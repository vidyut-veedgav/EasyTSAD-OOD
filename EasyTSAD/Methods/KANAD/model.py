from typing import Dict

import numpy as np
import torch as th
import torchinfo
import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from EasyTSAD.DataFactory import TSData
from EasyTSAD.DataFactory import MTSData
from EasyTSAD.Exptools import EarlyStoppingTorch
from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory.TorchDataSet import PredictWindow
from EasyTSAD.DataFactory.TorchDataSet.PredictWindow import UTSOneByOneDataset


class KANADModel(nn.Module):
    def __init__(self, window: int, order: int, *args, **kwargs) -> None:
        super().__init__()
        self.window = window
        self.order = order
        self.channels = 2 * self.order + 1
        self.register_buffer(
            "orders",
            self._create_custom_periodic_cosine().unsqueeze(0),  # (1, order, window)
        )
        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Conv1d(1, 1, window, padding=0, stride=1, dilation=1)

    def forward(self, x: th.Tensor, *args, **kwargs) -> th.Tensor:
        res = []
        res.append(x.unsqueeze(1))
        ff = th.concat(
            [self.orders.repeat(x.size(0), 1, 1)]
            + [th.cos(order * x.unsqueeze(1)) for order in range(1, self.order + 1)]
            + [x.unsqueeze(1)],
            dim=1,
        )  # (B, channels, window)
        res.append(ff)
        ff = self.init_conv(ff)
        ff = self.bn1(ff)
        ff = self.act(ff)
        ff = self.inner_conv(ff) + res.pop()
        ff = self.bn2(ff)
        ff = self.act(ff)
        ff = self.out_conv(ff) + res.pop()
        ff = self.bn3(ff)
        ff = self.act(ff)
        ff = self.final_conv(ff)
        return ff.squeeze(1)  # (B, 1)

    def _create_custom_periodic_cosine(self) -> th.Tensor:
        result = th.empty(self.order, self.window, dtype=th.float32)
        for i, p in enumerate(range(1, self.order + 1)):
            range_value = th.arange(self.window, dtype=th.float32)
            result[i, :] = th.cos(2 * th.pi * range_value * p / self.window)
        return result


class _MTSWindowDataset(Dataset):
    """Sliding-window dataset for MTS data. Returns (W, N) windows and (N,) targets."""
    def __init__(self, data: np.ndarray, window: int):
        self.data = th.from_numpy(data).float()  # (T, N)
        self.window = window

    def __len__(self):
        return max(len(self.data) - self.window, 0)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window]      # (W, N)
        y = self.data[idx + self.window]             # (N,)
        return x, y


class KANAD(BaseMethod):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.__anomaly_score = None

        if th.cuda.is_available():
            self.device = th.device("cuda")
            print("=== Using CUDA ===")
        else:
            self.device = th.device("cpu")
            print("=== Using CPU ===")

        self.batch_size = params["batch_size"]
        self.window = params["window"]
        self.debug = params.get("debug", False)
        self.model = KANADModel(**params).to(self.device)

        self.epochs = params["epochs"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=params["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.75)
        self.loss = nn.MSELoss()
        self.early_stopping = EarlyStoppingTorch(save_path=None, patience=3)

    # ------------------------------------------------------------------
    # UTS path
    # ------------------------------------------------------------------

    def _uts_train(self, tsTrain: TSData):
        train_loader = DataLoader(
            UTSOneByOneDataset(tsTrain, "train", window_size=self.window),
            batch_size=self.batch_size, shuffle=True,
        )
        valid_loader = DataLoader(
            UTSOneByOneDataset(tsTrain, "valid", window_size=self.window),
            batch_size=self.batch_size, shuffle=False,
        )
        self._run_epochs(train_loader, valid_loader)

    def _uts_test(self, tsData: TSData):
        loader = DataLoader(
            UTSOneByOneDataset(tsData, "test", window_size=self.window),
            batch_size=self.batch_size, shuffle=False,
        )
        self.model.eval()
        scores, all_x = [], []
        with th.no_grad():
            for x, target in tqdm.tqdm(loader, desc="Testing"):
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                scores.append(th.sub(output, target).abs().cpu())
                all_x.append(x.cpu())

        scores = th.cat(scores, dim=0)[..., -1].numpy().flatten()
        scores[np.isnan(scores)] = 1000
        if self.debug:
            th.save(self.model.state_dict(), "model.pth")
            np.save("all_x.npy", th.cat(all_x, dim=0)[..., -1].numpy())
        self.__anomaly_score = scores

    # ------------------------------------------------------------------
    # MTS path — channel-independent (paper §4.7)
    # Reshape (B, W, N) → (B*N, W), run through the same UTS model,
    # then average per-channel errors back to a single per-timestep score.
    # ------------------------------------------------------------------

    def _mts_train(self, tsTrain: MTSData):
        full = np.concatenate([tsTrain.train, tsTrain.valid], axis=0)
        n_total = len(full)
        split = int(n_total * 0.8)
        train_data, val_data = full[:split], full[split:]

        train_loader = DataLoader(
            _MTSWindowDataset(train_data, self.window),
            batch_size=self.batch_size, shuffle=True,
        )
        valid_loader = DataLoader(
            _MTSWindowDataset(val_data, self.window),
            batch_size=self.batch_size, shuffle=False,
        )
        self._run_epochs(train_loader, valid_loader, mts=True)

    def _mts_test(self, tsData: MTSData):
        loader = DataLoader(
            _MTSWindowDataset(tsData.test, self.window),
            batch_size=self.batch_size, shuffle=False,
        )
        self.model.eval()
        scores = []
        with th.no_grad():
            for x, target in tqdm.tqdm(loader, desc="Testing (MTS)"):
                # x: (B, W, N)  target: (B, N)
                B, W, N = x.shape
                x_flat = x.permute(0, 2, 1).reshape(B * N, W).to(self.device)   # (B*N, W)
                t_flat = target.reshape(B * N, 1).to(self.device)                # (B*N, 1)
                out = self.model(x_flat)                                          # (B*N, 1)
                err = th.abs(out - t_flat).reshape(B, N).mean(dim=-1)            # (B,)
                scores.append(err.cpu().numpy())

        scores = np.concatenate(scores)
        pad = np.full(self.window, scores.mean())
        self.__anomaly_score = np.concatenate([pad, scores])

    # ------------------------------------------------------------------
    # Shared epoch loop (UTS and MTS share the same model and optimiser)
    # ------------------------------------------------------------------

    def _run_epochs(self, train_loader, valid_loader, mts: bool = False):
        for epoch in range(1, self.epochs + 1):
            self.model.train(mode=True)
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, (x, target) in loop:
                if mts:
                    B, W, N = x.shape
                    x      = x.permute(0, 2, 1).reshape(B * N, W).to(self.device)
                    target = target.reshape(B * N, 1).to(self.device)
                else:
                    x, target = x.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.item()
                loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            self.model.eval()
            avg_loss = 0
            loop = tqdm.tqdm(enumerate(valid_loader), total=len(valid_loader), leave=True)
            with th.no_grad():
                for idx, (x, target) in loop:
                    if mts:
                        B, W, N = x.shape
                        x      = x.permute(0, 2, 1).reshape(B * N, W).to(self.device)
                        target = target.reshape(B * N, 1).to(self.device)
                    else:
                        x, target = x.to(self.device), target.to(self.device)

                    output = self.model(x)
                    loss = self.loss(output, target)
                    avg_loss += loss.item()
                    loop.set_description(f"Validation Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

            valid_loss = avg_loss / max(len(valid_loader), 1)
            self.scheduler.step()
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break

    # ------------------------------------------------------------------
    # BaseMethod interface — dispatches on data dimensionality
    # ------------------------------------------------------------------

    def train_valid_phase(self, tsTrain):
        if tsTrain.train.ndim == 2:
            self._mts_train(tsTrain)
        else:
            self._uts_train(tsTrain)

    def test_phase(self, tsData):
        if tsData.test.ndim == 2:
            self._mts_test(tsData)
        else:
            self._uts_test(tsData)

    def train_valid_phase_all_in_one(self, tsTrains: Dict[str, TSData]):
        train_loader = DataLoader(
            PredictWindow.UTSAllInOneDataset(tsTrains, "train", window_size=self.window),
            batch_size=self.batch_size, shuffle=True,
        )
        valid_loader = DataLoader(
            PredictWindow.UTSAllInOneDataset(tsTrains, "valid", window_size=self.window),
            batch_size=self.batch_size, shuffle=False,
        )
        self._run_epochs(train_loader, valid_loader)

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score  # type: ignore

    def param_statistic(self, save_file):
        model_stats = torchinfo.summary(self.model, (self.batch_size, self.window), verbose=0)
        with open(save_file, "w", encoding="utf-8") as f:
            f.write(str(model_stats))
