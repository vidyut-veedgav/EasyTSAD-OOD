import os
import sys
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

_SUBMODULE_PATH = os.path.abspath(os.path.dirname(__file__))
if _SUBMODULE_PATH not in sys.path:
    sys.path.insert(0, _SUBMODULE_PATH)

# Import only the pure CATCH model/loss files — avoids CATCH.py which pulls in
# ts_benchmark.baselines.utils → time_series_library → reformer_pytorch.
from ts_benchmark.baselines.catch.models.CATCH_model import CATCHModel
from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_loss, frequency_criterion
from ts_benchmark.baselines.catch.utils.tools import EarlyStopping, adjust_learning_rate

from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import MTSData


DEFAULT_HYPER_PARAMS = {
    'seq_len':               192,
    'patch_size':            16,
    'patch_stride':          8,
    'e_layers':              3,
    'n_heads':               2,
    'd_model':               128,
    'cf_dim':                64,
    'd_ff':                  256,
    'head_dim':              64,
    'dropout':               0.2,
    'head_dropout':          0.1,
    'individual':            0,
    'revin':                 1,
    'affine':                0,
    'subtract_last':         0,
    'regular_lambda':        0.5,
    'temperature':           0.07,
    'auxi_loss':             'MAE',
    'auxi_type':             'complex',
    'auxi_mode':             'fft',
    'auxi_lambda':           0.005,
    'dc_lambda':             0.005,
    'score_lambda':          0.05,
    'lr':                    0.0001,
    'Mlr':                   0.00001,
    'pct_start':             0.3,
    'lradj':                 'type1',
    'num_epochs':            3,
    'batch_size':            128,
    'patience':              3,
    'module_first':          True,
    'mask':                  False,
    'inference_patch_size':  32,
    'inference_patch_stride': 1,
}


class TransformerConfig:
    """Attribute bag matching CATCH.py's TransformerConfig — defined inline to avoid CATCH.py's import chain."""
    def __init__(self, **kwargs):
        for k, v in DEFAULT_HYPER_PARAMS.items():
            setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def pred_len(self):
        return self.seq_len

    @property
    def learning_rate(self):
        return self.lr


class _SegDataset(Dataset):
    """Sliding-window dataset mirroring CATCH's SegLoader, but without the heavy utils import."""
    def __init__(self, data: np.ndarray, win_size: int, step: int, mode: str = "train"):
        self.data = data
        self.win_size = win_size
        self.step = step
        self.mode = mode

    def __len__(self):
        if self.mode == "thre":
            return (self.data.shape[0] - self.win_size) // self.win_size + 1
        return (self.data.shape[0] - self.win_size) // self.step + 1

    def __getitem__(self, index):
        if self.mode == "thre":
            start = index * self.win_size
        else:
            start = index * self.step
        x = np.float32(self.data[start : start + self.win_size])
        return x, x  # (window, N), dummy label


def _make_loader(data: np.ndarray, win_size: int, batch_size: int, mode: str) -> DataLoader:
    step = 1
    shuffle = mode in ("train", "val")
    ds = _SegDataset(data, win_size, step, mode)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def _build_config(n_vars: int, params: dict) -> TransformerConfig:
    merged = {**DEFAULT_HYPER_PARAMS, **params}
    config = TransformerConfig(**merged)
    config.enc_in    = n_vars
    config.c_in      = n_vars
    config.c_out     = n_vars
    config.label_len = 48
    config.task_name = "anomaly_detection"
    return config


class CATCH(BaseMethod):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        self._params = params
        self._model = None
        self._config = None
        self._scaler = StandardScaler()
        self._best_state = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_valid_phase(self, tsTrain: MTSData):
        full_train = np.concatenate([tsTrain.train, tsTrain.valid], axis=0)
        n_vars = full_train.shape[1]

        # 80/20 split (matches CATCH paper)
        split = int(len(full_train) * 0.8)
        train_data, val_data = full_train[:split], full_train[split:]

        self._scaler.fit(train_data)
        train_data = self._scaler.transform(train_data)
        val_data   = self._scaler.transform(val_data)

        config = _build_config(n_vars, self._params)
        self._config = config
        model = CATCHModel(config).to(self.device)
        self._model = model

        batch_size = config.batch_size
        train_loader = _make_loader(train_data, config.seq_len, batch_size, "train")
        val_loader   = _make_loader(val_data,   config.seq_len, batch_size, "val")

        criterion  = nn.MSELoss()
        auxi_loss  = frequency_loss(config)
        main_params = [p for n, p in model.named_parameters() if "mask_generator" not in n]
        optimizer  = torch.optim.Adam(main_params, lr=config.lr)
        optimizerM = torch.optim.Adam(model.mask_generator.parameters(), lr=config.Mlr)

        train_steps = len(train_loader)
        scheduler  = lr_scheduler.OneCycleLR(optimizer,  steps_per_epoch=train_steps,
                                              pct_start=config.pct_start, epochs=config.num_epochs, max_lr=config.lr)
        schedulerM = lr_scheduler.OneCycleLR(optimizerM, steps_per_epoch=train_steps,
                                              pct_start=config.pct_start, epochs=config.num_epochs, max_lr=config.Mlr)

        early_stopping = EarlyStopping(patience=config.patience, verbose=True)

        for epoch in range(config.num_epochs):
            model.train()
            train_losses = []
            for i, (batch_x, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                optimizer.zero_grad()
                output, output_complex, dcloss = model(batch_x)
                rec_loss  = criterion(output, batch_x)
                norm_input = model.revin_layer(batch_x, "transform")
                aux_loss  = auxi_loss(output_complex, norm_input)
                loss = rec_loss + config.dc_lambda * dcloss + config.auxi_lambda * aux_loss
                loss.backward()
                optimizer.step()
                if (i + 1) % min(len(train_loader) // 10 or 1, 100) == 0:
                    optimizerM.step()
                    optimizerM.zero_grad()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.float().to(self.device)
                    output, _, _ = model(batch_x)
                    val_losses.append(criterion(output, batch_x).item())
            val_loss = float(np.mean(val_losses))
            print(f"Epoch {epoch+1}/{config.num_epochs} | train: {np.mean(train_losses):.6f} | val: {val_loss:.6f}")

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer,  scheduler,  epoch + 1, config)
            adjust_learning_rate(optimizerM, schedulerM, epoch + 1, config, printout=False)

        self._best_state = copy.deepcopy(early_stopping.check_point)

    def test_phase(self, tsData: MTSData):
        test_data = self._scaler.transform(tsData.test)
        config = self._config
        self._model.load_state_dict(self._best_state)
        self._model.eval().to(self.device)

        loader = _make_loader(test_data, config.seq_len, config.batch_size, "thre")
        temp_crit = nn.MSELoss(reduction="none")
        freq_crit = frequency_criterion(config)

        scores = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.float().to(self.device)
                output, _, _ = self._model(batch_x)
                temp_score = torch.mean(temp_crit(batch_x, output), dim=-1)
                freq_score = torch.mean(freq_crit(batch_x, output),  dim=-1)
                score = (temp_score + config.score_lambda * freq_score).cpu().numpy()
                scores.append(score)

        self.__anomaly_score = np.concatenate(scores).flatten()

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        if self._model is not None:
            total = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            with open(save_file, "w") as f:
                f.write(f"Total trainable parameters: {total}")
