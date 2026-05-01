import math
from typing import Optional

import einops
import numpy as np
import torch as th
import torchinfo
import tqdm
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader, Dataset

from EasyTSAD.DataFactory import TSData, MTSData
from EasyTSAD.Exptools import EarlyStoppingTorch
from EasyTSAD.Methods import BaseMethod


# ============================================================
# SAR73 / SAR76 neural network components
# (inlined from ood-quantcwru/methods/sarad/external/src/models/components/)
# ============================================================

class _SpatialEncoding(nn.Module):
    def __init__(self, input_size: int, model_size: int):
        super().__init__()
        self.se = nn.Parameter(th.randn(1, 1, input_size, model_size))

    def forward(self) -> Tensor:
        return self.se


class _Embedding(nn.Module):
    def __init__(self, input_size: int, patch_size: int, model_size: int, dropout: float):
        super().__init__()
        self.encoding = nn.Linear(patch_size, model_size)
        self.spatial_encoding = _SpatialEncoding(input_size, model_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, n_patches, input_size, patch_size)
        return self.dropout(self.encoding(x) + self.spatial_encoding())


class _Patching(nn.Module):
    def __init__(self, num_patches: int):
        super().__init__()
        self.num_patches = num_patches

    def forward(self, x: Tensor) -> Tensor:
        # (B, W, N) -> (B, n_patches, N, patch_size)
        return einops.rearrange(x, 'b (p s) i -> b p i s', p=self.num_patches)


class _Unpatching(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        # (B, n_patches, N, patch_size) -> (B, W, N)
        return einops.rearrange(x, 'b p i s -> b (p s) i')


class _Attention(nn.Module):
    def __init__(self, input_size: int, model_size: int, n_heads: int,
                 dropout: float, bias: bool, is_diagonal_masked: bool):
        super().__init__()
        assert model_size % n_heads == 0
        self.model_size = model_size
        self.n_heads = n_heads
        self.head_size = model_size // n_heads
        self.is_diagonal_masked = is_diagonal_masked

        self.Q = nn.Linear(model_size, model_size, bias)
        self.K = nn.Linear(model_size, model_size, bias)
        self.V = nn.Linear(model_size, model_size, bias)
        self.linear = nn.Linear(model_size, model_size)
        self.dropout = nn.Dropout(dropout)

        diag_mask = 1.0 - th.eye(input_size).unsqueeze(0).unsqueeze(0)
        self.register_buffer('diag_mask', diag_mask)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, s: Optional[Tensor] = None):
        B, N, _ = q.size()
        v = self.V(v).view(B, N, self.n_heads, self.head_size)
        if s is None:
            q = self.Q(q).view(B, N, self.n_heads, self.head_size)
            k = self.K(k).view(B, N, self.n_heads, self.head_size)
            scores = th.einsum("bqhe,bkhe->bhqk", [q, k]) / math.sqrt(self.head_size)
            s = th.softmax(scores, dim=-1)
            if self.is_diagonal_masked:
                s = s * self.diag_mask
                s = s / (s.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            if self.is_diagonal_masked:
                s = s * self.diag_mask
            s = s - s.min(dim=1, keepdim=True)[0]
            s = s / (s.sum(dim=-1, keepdim=True) + 1e-6)

        attention = th.einsum("bhql,blhd->bqhd", [self.dropout(s), v])
        attention = self.linear(attention.reshape(B, N, self.model_size))
        return attention, s


class _SpatialEncoder(nn.Module):
    def __init__(self, input_size: int, model_size: int, feedforward_size: int,
                 num_heads: int, dropout: float, bias: bool, is_diagonal_masked: bool):
        super().__init__()
        self.num_heads = num_heads
        self.attention = _Attention(input_size, model_size, num_heads, dropout,
                                    bias, is_diagonal_masked)
        self.norm1 = nn.LayerNorm(model_size)
        self.norm2 = nn.LayerNorm(model_size)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_size, feedforward_size, bias), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward_size, model_size, bias), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, s: Optional[Tensor] = None):
        if x.dim() == 4:
            B, P, N, M = x.size()
            z = x.view(B * P, N, M)
        else:
            z = x
        attention, s = self.attention(z, z, z, s)
        z = self.norm1(z + self.dropout(attention))
        z = self.norm2(z + self.feed_forward(z))
        if x.dim() == 4:
            z = z.view(B, P, N, M)
            s = s.view(B, P, self.num_heads, N, N)
        return z, s


class _Decoder(nn.Module):
    def __init__(self, patch_size: int, model_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(model_size)
        self.linear = nn.Linear(model_size, patch_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.ln(x))


class SAR76Model(nn.Module):
    def __init__(self, input_size: int, window_size: int, model_size: int,
                 num_layers: int, num_heads: int, num_patches: int,
                 detector_size: int, dropout: float, is_diagonal_masked: bool = False):
        assert num_patches == 2, 'num_patches must be 2'
        super().__init__()
        patch_size = window_size // num_patches
        feedforward_size = 4 * model_size

        self.patching = _Patching(num_patches)
        self.embedding = _Embedding(input_size, patch_size, model_size, dropout)
        self.encoders = nn.ModuleList([
            _SpatialEncoder(input_size, model_size, feedforward_size, num_heads,
                            dropout, bias=True, is_diagonal_masked=is_diagonal_masked)
            for _ in range(num_layers)
        ])
        self.decoder = _Decoder(patch_size, model_size)
        self.unpatching = _Unpatching()
        self.detector = nn.Sequential(
            nn.Linear(num_heads * input_size, detector_size), nn.ReLU(),
            nn.Linear(detector_size, num_heads * input_size), nn.ReLU(),
        )

    def forward(self, x: Tensor):
        # x: (B, W, N)
        z = self.embedding(self.patching(x))
        s_all = []
        for encoder in self.encoders:
            z, s_layer = encoder(z)
            s_all.append(s_layer)
        s_all = th.stack(s_all, dim=1)            # (B, L, P, H, N, N)
        x_hat = self.unpatching(self.decoder(z))  # (B, W, N)

        s_det = s_all.detach()
        q = th.relu(s_det[:, :, 0] - s_det[:, :, 1])[:, -1].sum(-2)  # (B, H, N)
        q_bar = self.detector(q.flatten(1)).reshape_as(q)              # (B, H, N)
        return x_hat, s_all, q, q_bar


# ============================================================
# Sliding-window reconstruction dataset
# ============================================================

class _ReconWindowDataset(Dataset):
    """Sliding-window dataset for reconstruction. Each item is one window (W, N)."""
    def __init__(self, data: np.ndarray, window: int):
        self.data = th.from_numpy(data).float()
        self.window = window

    def __len__(self):
        return max(len(self.data) - self.window + 1, 0)

    def __getitem__(self, idx):
        return self.data[idx: idx + self.window]  # (W, N)


# ============================================================
# EasyTSAD BaseMethod wrapper
# ============================================================

class SARAD(BaseMethod):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.__anomaly_score = None

        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        print(f"=== Using {'CUDA' if self.device.type == 'cuda' else 'CPU'} ===")

        self.window = params["window"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.lr = params.get("lr", 7e-4)
        self.detec_weight = params.get("detec_weight", 100.0)

        self._net_kwargs = {
            "window_size": self.window,
            "model_size": params.get("model_size", 64),
            "num_layers": params.get("num_layers", 3),
            "num_heads": params.get("num_heads", 8),
            "num_patches": params.get("num_patches", 2),
            "detector_size": params.get("detector_size", 64),
            "dropout": params.get("dropout", 0.1),
            "is_diagonal_masked": params.get("is_diagonal_masked", False),
        }

        # model is built lazily once input_size is known from data
        self.model: SAR76Model = None
        self.criterion = nn.MSELoss(reduction="none")

        # normalization stats updated each validation epoch
        self._recon_avg = 0.0
        self._recon_std = 1.0
        self._detec_avg = 0.0
        self._detec_std = 1.0

    def _build_model(self, input_size: int):
        self.model = SAR76Model(input_size=input_size, **self._net_kwargs).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)
        self.early_stopping = EarlyStoppingTorch(save_path=None, patience=3)

    def _forward_losses(self, x: th.Tensor):
        """Returns per-sample (recon_loss, detec_loss), both shape (B,)."""
        x_hat, _, q, q_bar = self.model(x)
        recon_loss = self.criterion(x_hat, x).mean(dim=(1, 2))
        detec_loss = self.criterion(q, q_bar).mean(dim=tuple(range(1, q.dim())))
        return recon_loss, detec_loss

    def _make_loaders(self, data: np.ndarray):
        split = int(len(data) * 0.8)
        train_ds = _ReconWindowDataset(data[:split], self.window)
        valid_ds = _ReconWindowDataset(data[split:], self.window)
        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True),
            DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False),
        )

    def _run_epochs(self, train_loader, valid_loader):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            avg_loss = 0.0
            loop = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
            for idx, x in loop:
                x = x.to(self.device)
                self.optimizer.zero_grad()
                recon_loss, detec_loss = self._forward_losses(x)
                loss = recon_loss.mean() + self.detec_weight * detec_loss.mean()
                loss.backward()
                self.optimizer.step()
                avg_loss += loss.item()
                loop.set_description(f"Train [{epoch}/{self.epochs}]")
                loop.set_postfix(loss=loss.item(), avg=avg_loss / (idx + 1))

            self.model.eval()
            val_loss = 0.0
            all_recon, all_detec = [], []
            with th.no_grad():
                for x in valid_loader:
                    x = x.to(self.device)
                    recon_loss, detec_loss = self._forward_losses(x)
                    all_recon.append(recon_loss.cpu())
                    all_detec.append(detec_loss.cpu())
                    val_loss += (recon_loss.mean() + self.detec_weight * detec_loss.mean()).item()

            recon_arr = th.cat(all_recon).numpy()
            detec_arr = th.cat(all_detec).numpy()
            self._recon_avg = float(recon_arr.mean())
            self._recon_std = float(max(recon_arr.std(), 1e-6))
            self._detec_avg = float(detec_arr.mean())
            self._detec_std = float(max(detec_arr.std(), 1e-6))

            val_loss /= max(len(valid_loader), 1)
            self.scheduler.step()
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("   Early stopping<<<")
                break

    def _score(self, loader: DataLoader, total_len: int):
        self.model.eval()
        scores = []
        with th.no_grad():
            for x in tqdm.tqdm(loader, desc="Testing"):
                x = x.to(self.device)
                recon_loss, detec_loss = self._forward_losses(x)
                score = (recon_loss.cpu().numpy() - self._recon_avg) / self._recon_std + \
                        (detec_loss.cpu().numpy() - self._detec_avg) / self._detec_std
                scores.append(score)

        scores = np.concatenate(scores)
        pad_len = total_len - len(scores)
        pad = np.full(pad_len, scores.mean()) if pad_len > 0 else np.array([])
        self.__anomaly_score = np.concatenate([pad, scores])

    # ------------------------------------------------------------------
    # UTS path — treat (T,) as single-channel MTS (T, 1)
    # ------------------------------------------------------------------

    def _uts_train(self, tsTrain: TSData):
        data = np.concatenate([tsTrain.train, tsTrain.valid]).reshape(-1, 1)
        self._build_model(input_size=1)
        self._run_epochs(*self._make_loaders(data))

    def _uts_test(self, tsData: TSData):
        data = tsData.test.reshape(-1, 1)
        loader = DataLoader(_ReconWindowDataset(data, self.window),
                            batch_size=self.batch_size, shuffle=False)
        self._score(loader, len(data))

    # ------------------------------------------------------------------
    # MTS path
    # ------------------------------------------------------------------

    def _mts_train(self, tsTrain: MTSData):
        data = np.concatenate([tsTrain.train, tsTrain.valid], axis=0)
        self._build_model(input_size=data.shape[1])
        self._run_epochs(*self._make_loaders(data))

    def _mts_test(self, tsData: MTSData):
        data = tsData.test
        loader = DataLoader(_ReconWindowDataset(data, self.window),
                            batch_size=self.batch_size, shuffle=False)
        self._score(loader, len(data))

    # ------------------------------------------------------------------
    # BaseMethod interface
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

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score  # type: ignore

    def param_statistic(self, save_file):
        if self.model is None:
            return
        N = self._net_kwargs.get("input_size", 1)
        stats = torchinfo.summary(
            self.model, (self.batch_size, self.window, N), verbose=0
        )
        with open(save_file, "w") as f:
            f.write(str(stats))
