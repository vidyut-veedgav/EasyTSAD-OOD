import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset

from EasyTSAD.Methods import BaseMethod
from EasyTSAD.DataFactory import MTSData
from EasyTSAD.Exptools import EarlyStoppingTorch

class FrequencyDecomposition(nn.Module):
    def __init__(self, T_len: int, top_m_percent: int = 20):
        super().__init__()
        self.T_len = T_len
        self.top_m_percent = top_m_percent

    def forward(self, x_time_domain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, N = x_time_domain.shape
        x_fft = torch.fft.rfft(x_time_domain, dim=1, norm="forward")
        amplitudes = torch.abs(x_fft)
        
        avg_amp_per_bin = amplitudes.mean(dim=(0, 2))
        _, sorted_indices_global = torch.sort(avg_amp_per_bin, descending=True)
        num_freq_bins = amplitudes.shape[1]
        num_top_freq = int(num_freq_bins * (self.top_m_percent / 100))
        time_invariant_indices = sorted_indices_global[:num_top_freq]

        mask_inv = torch.zeros(num_freq_bins, dtype=torch.bool, device=x_fft.device)
        mask_inv[time_invariant_indices] = True
        
        x_fft_inv = x_fft * mask_inv.unsqueeze(0).unsqueeze(-1)
        x_fft_var = x_fft * (~mask_inv).unsqueeze(0).unsqueeze(-1)
        x_inv = torch.fft.irfft(x_fft_inv, n=T, dim=1, norm="forward")
        x_var = torch.fft.irfft(x_fft_var, n=T, dim=1, norm="forward")
        
        return x_inv, x_var

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class StructuralProtoBank(nn.Module): 
    def __init__(self, num_protos: int, num_vars: int):
        super().__init__()
        self.prototype_masks = nn.Parameter(torch.randn(num_protos, num_vars, num_vars))

    def forward(self, selector_logits: torch.Tensor, tau: float = 1.0, hard: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        soft_weights = F.gumbel_softmax(selector_logits, tau=tau, hard=hard, dim=-1)
        weighted_mask = torch.einsum("bk,kij->bij", soft_weights, self.prototype_masks)
        return weighted_mask, soft_weights

class ProtoMaskSelector(nn.Module):
    def __init__(self, d_model: int, num_protos: int):
        super().__init__()
        self.fc = nn.Linear(d_model, num_protos)

    def forward(self, global_feat: torch.Tensor) -> torch.Tensor:
        return self.fc(global_feat)

class ContextProtoBank(nn.Module):
    def __init__(self, num_context_protos: int, d_model: int):
        super().__init__()
        self.context_prototypes = nn.Parameter(torch.randn(num_context_protos, d_model))
        self.context_selector = nn.Linear(d_model, num_context_protos)

    def forward(self, global_feat: torch.Tensor, tau: float = 1.0, hard: bool = False) -> tuple:
        selector_logits = self.context_selector(global_feat)
        soft_weights = F.gumbel_softmax(selector_logits, tau=tau, hard=hard, dim=-1)
        selected_context_protos = torch.einsum("bk,kd->bd", soft_weights, self.context_prototypes)
        
        dot_product = (global_feat * selected_context_protos).sum(dim=-1)
        norm_global_feat = torch.norm(global_feat, dim=-1)
        norm_selected_protos = torch.norm(selected_context_protos, dim=-1)
        cosine_sim = dot_product / ((norm_global_feat * norm_selected_protos) + 1e-8)
        deviation_score = 1 - cosine_sim
        
        return deviation_score, selected_context_protos, soft_weights

class SpikeProtoBank(nn.Module):
    def __init__(self, num_spike_protos: int, d_model: int, seq_len: int):
        super().__init__()
        self.spike_prototypes = nn.Parameter(torch.randn(num_spike_protos, d_model))
        self.spike_query_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.spike_query_mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, H: torch.Tensor, tau: float = 1.0, hard: bool = False) -> tuple:
        H_permuted = H.permute(0, 2, 1)
        conv_output = F.relu(self.spike_query_conv(H_permuted))
        q_c_spike = F.max_pool1d(conv_output, kernel_size=conv_output.shape[2]).squeeze(-1)
        q_c_spike_final = self.spike_query_mlp(q_c_spike)
        
        distances = torch.cdist(q_c_spike_final, self.spike_prototypes)
        deviation_score = torch.min(distances, dim=-1).values
        
        return deviation_score, q_c_spike_final

class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, cnn_kernel_size: int = 3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.conv_module = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, kernel_size=1, stride=1), nn.GLU(dim=1),
            nn.Conv1d(d_model, d_model, kernel_size=cnn_kernel_size, padding=(cnn_kernel_size - 1) // 2),
            nn.BatchNorm1d(d_model), nn.SiLU(),
            nn.Conv1d(d_model, d_model, kernel_size=1, stride=1), nn.Dropout(dropout)
        )
        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None, **kwargs) -> torch.Tensor:
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src = src + self.conv_module(src.permute(0, 2, 1)).permute(0, 2, 1)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm3(src)
        return src

# --- Main Model: PGRFNet ---

class PGRFNet(nn.Module):
    def __init__(self, num_vars: int, seq_len: int, num_protos: int, num_context_protos: int, num_spike_protos: int,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 2,
                 dim_ff: int = 256, top_m_percent: int = 20):
        super().__init__()
        self.num_vars, self.seq_len, self.d_model = num_vars, seq_len, d_model

        # Stage 1: Multi-faceted Evidence Extractor
        self.freq_decomposition = FrequencyDecomposition(seq_len, top_m_percent)
        self.time_invariant_embed = nn.Linear(num_vars, d_model)
        self.time_variant_embed = nn.Linear(num_vars, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 10)
        self.time_invariant_encoder = nn.TransformerEncoder(ConformerBlock(d_model, nhead, dim_ff), num_layers, enable_nested_tensor=False)
        self.time_variant_encoder = nn.TransformerEncoder(ConformerBlock(d_model, nhead, dim_ff), num_layers, enable_nested_tensor=False)
        self.fusion_linear = nn.Linear(2 * d_model, d_model)
        
        self.mask_selector = ProtoMaskSelector(d_model, num_protos)
        self.proto_bank = StructuralProtoBank(num_protos, num_vars)
        
        self.context_proto_bank = ContextProtoBank(num_context_protos, d_model)
        self.spike_proto_bank = SpikeProtoBank(num_spike_protos, d_model, seq_len)
        self.predictors = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1)) for _ in range(num_vars)
        ])
        self.register_buffer('base_structural_mask', torch.zeros(num_vars, num_vars))
        self.register_buffer('is_base_mask_set', torch.tensor(False))
        
        # Stage 2: Gated Evidence Fusion Network
        self.gate_controller = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 4)
        )

    def forward(self, x_orig: torch.Tensor, return_internals: bool = False) -> tuple:
        B, T, N = x_orig.shape
        x_inv, x_var = self.freq_decomposition(x_orig)
        h_inv = self.time_invariant_encoder(self.pos_enc(self.time_invariant_embed(x_inv)))
        h_var = self.time_variant_encoder(self.pos_enc(self.time_variant_embed(x_var)))
        H = self.fusion_linear(torch.cat((h_inv, h_var), dim=-1))
        global_feat = H.mean(dim=1)

        selector_logits = self.mask_selector(global_feat)
        dynamic_mask_per_batch, mask_proto_weights = self.proto_bank(selector_logits)
        structural_mask_for_penalty = dynamic_mask_per_batch.mean(dim=0) 

        preds = []
        feat_expanded_for_graph = global_feat.unsqueeze(1).repeat(1, N, 1)
        for i in range(N):
            parent_feat = torch.einsum("bi,bid->bd", dynamic_mask_per_batch[:, i, :], feat_expanded_for_graph)
            preds.append(self.predictors[i](parent_feat))

        context_deviation_score, _, ctx_proto_weights = self.context_proto_bank(global_feat)
        
        spike_deviation_score, _ = self.spike_proto_bank(h_var)
        
        raw_gate_values = self.gate_controller(global_feat)
        gate_weights = F.softmax(raw_gate_values, dim=-1)

        if self.training and not self.is_base_mask_set.item() and B > 0:
            self.base_structural_mask.copy_(structural_mask_for_penalty.detach())
            self.is_base_mask_set.fill_(True)

        outputs = [
            torch.cat(preds, dim=1), dynamic_mask_per_batch, structural_mask_for_penalty,
            context_deviation_score, spike_deviation_score, raw_gate_values, gate_weights
        ]

        if return_internals:
            outputs.append({
                "mask_weights": mask_proto_weights,
                "context_weights": ctx_proto_weights,
                "dynamic_mask": dynamic_mask_per_batch
            })
        return tuple(outputs)


class _PGRFDataset(Dataset):
    def __init__(self, data: np.ndarray, window: int):
        self.data = torch.from_numpy(data).float()
        self.window = window

    def __len__(self):
        return max(len(self.data) - self.window, 0)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window]       # (window, N)
        y = self.data[idx + self.window]              # (N,)
        return x, y


class PGRF(BaseMethod):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.__anomaly_score = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.window = params["window"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.lr = params["lr"]
        self._params = params
        self.model = None  # built in train_valid_phase once num_vars is known

    def _build_model(self, num_vars: int) -> PGRFNet:
        p = self._params
        return PGRFNet(
            num_vars=num_vars,
            seq_len=self.window,
            num_protos=p.get("num_protos", 8),
            num_context_protos=p.get("num_context_protos", 8),
            num_spike_protos=p.get("num_spike_protos", 8),
            d_model=p.get("d_model", 128),
            nhead=p.get("nhead", 4),
            num_layers=p.get("num_layers", 2),
            dim_ff=p.get("dim_ff", 256),
            top_m_percent=p.get("top_m_percent", 20),
        ).to(self.device)

    def train_valid_phase(self, tsTrain: MTSData):
        import copy
        num_vars = tsTrain.train.shape[1]
        self.model = self._build_model(num_vars)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        early_stopping = EarlyStoppingTorch(save_path=None, patience=3)

        train_loader = DataLoader(
            _PGRFDataset(tsTrain.train, self.window),
            batch_size=self.batch_size, shuffle=True,
        )
        valid_loader = DataLoader(
            _PGRFDataset(tsTrain.valid, self.window),
            batch_size=self.batch_size, shuffle=False,
        )

        best_state = None
        best_val_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                preds, *_ = self.model(x)
                loss_fn(preds, y).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in valid_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    preds, *_ = self.model(x)
                    val_losses.append(loss_fn(preds, y).item())
            val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            print(f"Epoch {epoch}/{self.epochs} | val_loss: {val_loss:.6f}")

            if np.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def test_phase(self, tsData: MTSData):
        self.model.eval()
        loader = DataLoader(
            _PGRFDataset(tsData.test, self.window),
            batch_size=self.batch_size, shuffle=False,
        )
        scores = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                preds, _, _, ctx_dev, spike_dev, _, gate_w = self.model(x)
                pred_err = torch.abs(preds - y).mean(dim=-1)          # (B,)
                score = (gate_w[:, 0] * pred_err
                         + gate_w[:, 1] * ctx_dev
                         + gate_w[:, 2] * spike_dev)
                scores.append(score.cpu().numpy())

        scores = np.concatenate(scores)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        pad = np.full(self.window, scores.mean())
        self.__anomaly_score = np.concatenate([pad, scores])

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        if self.model is not None:
            import torchinfo
            stats = torchinfo.summary(
                self.model, (self.batch_size, self.window, self.model.num_vars), verbose=0
            )
            with open(save_file, "w") as f:
                f.write(str(stats))
