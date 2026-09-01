"""GPT decoder-only của Luna Zero — kiến trúc cỡ GPT-2 small, tiếng Việt.

Không nạp trọng số của ai. Mọi tham số ở đây khởi tạo ngẫu nhiên rồi train từ số 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from luna_zero.config import MODEL, ModelConfig

# Phép tính kích cỡ sống ở sizing.py (không cần torch). Re-export để mã cũ không gãy.
from luna_zero.sizing import (
    ACTIVATION_FACTOR,
    BYTES_FP16,
    BYTES_PER_PARAM_ADAMW,
    estimate_num_params,
    estimate_vram_gb,
    format_params,
)

__all__ = [
    "ACTIVATION_FACTOR",
    "BYTES_FP16",
    "BYTES_PER_PARAM_ADAMW",
    "Block",
    "CausalSelfAttention",
    "GPTOutput",
    "LunaZeroGPT",
    "MLP",
    "estimate_num_params",
    "estimate_vram_gb",
    "format_params",
]


# --- mạng -------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Self-attention nhân quả: vị trí t chỉ được nhìn các vị trí <= t.

    Dùng `scaled_dot_product_attention(is_causal=True)` của PyTorch thay vì tự dựng mặt
    nạ. Hai lý do: nó gọi nhân flash-attention nên không vật chất hoá ma trận T x T
    (1024x1024 mỗi head mỗi lớp sẽ ngốn VRAM vô ích), và mặt nạ do thư viện lo thì
    không có cơ hội viết sai chiều — lỗi rò rỉ tương lai cho loss ĐẸP GIẢ, rất khó phát hiện.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_head != 0:
            raise ValueError("d_model phải chia hết cho n_head")
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.dropout = cfg.dropout
        # Gộp q, k, v vào một phép nhân: ít kernel launch hơn, nhanh hơn rõ rệt.
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x), approximate="tanh")))


class Block(nn.Module):
    """Pre-LayerNorm: chuẩn hoá TRƯỚC mỗi nhánh, cộng residual sau.

    Khác GPT-1 gốc (post-LN). Pre-LN cho đường residual sạch từ đầu tới cuối nên
    gradient không bị nhiễu qua 12 lớp — train ổn định hơn hẳn, không cần warmup dài.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


@dataclass
class GPTOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None


class LunaZeroGPT(nn.Module):
    def __init__(self, cfg: ModelConfig = MODEL) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(cfg.vocab_size, cfg.d_model),
                "wpe": nn.Embedding(cfg.block_size, cfg.d_model),
                "drop": nn.Dropout(cfg.dropout),
                "h": nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
                "ln_f": nn.LayerNorm(cfg.d_model, bias=cfg.bias),
            }
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Buộc chung trọng số embedding vào và lớp ra. Tiết kiệm 24,6M tham số
        # (22% của model) và thường cho kết quả tốt hơn ở quy mô này.
        self.transformer["wte"].weight = self.lm_head.weight

        self.apply(self._init_weights)
        # Khởi tạo co lại cho các phép chiếu ra khỏi residual: phương sai của đường
        # residual cộng dồn qua n_layer lớp, không co thì tín hiệu phình lên theo sqrt(n).
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def so_tham_so(self, tru_pos_emb: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if tru_pos_emb:
            n -= self.transformer["wpe"].weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> GPTOutput:
        _, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"chuỗi dài {T} vượt block_size {self.cfg.block_size}")
        pos = torch.arange(T, device=idx.device)

        x = self.transformer["drop"](self.transformer["wte"](idx) + self.transformer["wpe"](pos))
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return GPTOutput(logits=logits, loss=loss)

    @torch.no_grad()
    def sinh(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        phat_lap: float = 1.0,
    ) -> torch.Tensor:
        """Sinh token tự hồi quy. Cắt ngữ cảnh về block_size khi vượt.

        `top_p` (nucleus): giữ nhóm token nhỏ nhất có tổng xác suất >= top_p. Tốt hơn
        top_k vì kích thước nhóm co giãn theo độ chắc chắn của model — chỗ nào model
        chắc thì chọn hẹp, chỗ nào mơ hồ thì mở rộng. top_k cố định 50 sẽ ép chọn trong
        50 token kể cả khi model gần như chắc chắn, đó là một nguồn sinh vòng lặp.

        `phat_lap` > 1 hạ điểm các token ĐÃ xuất hiện. Model 110M non rất hay rơi vào
        vòng lặp kiểu "bảo đảm, bảo đảm, bảo đảm" — phạt lặp cắt vòng đó. Lưu ý: đây là
        che triệu chứng, không phải chữa. Cách chữa thật là train thêm.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits = self(idx_cond).logits[:, -1, :]

            if phat_lap != 1.0:
                for b in range(idx.size(0)):
                    da_co = torch.unique(idx_cond[b])
                    diem = logits[b, da_co]
                    # Điểm âm phải NHÂN lên (càng âm hơn), điểm dương phải CHIA xuống.
                    # Chia đều cho mọi điểm sẽ làm điểm âm to lên và thành thưởng lặp.
                    logits[b, da_co] = torch.where(diem < 0, diem * phat_lap, diem / phat_lap)

            logits = logits / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            if top_p is not None:
                sap, chi_so = torch.sort(logits, descending=True, dim=-1)
                tich_luy = torch.cumsum(F.softmax(sap, dim=-1), dim=-1)
                bo = tich_luy - F.softmax(sap, dim=-1) >= top_p
                bo[:, 0] = False  # luôn giữ token có xác suất cao nhất
                sap = sap.masked_fill(bo, -float("inf"))
                logits = torch.full_like(logits, -float("inf")).scatter(1, chi_so, sap)

            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx

    def nhom_tham_so_optimizer(self, weight_decay: float) -> list[dict]:
        """Chỉ phạt weight decay lên ma trận, KHÔNG lên bias và LayerNorm.

        Phạt LayerNorm là kéo hệ số chuẩn hoá về 0 — làm hỏng chính thứ giữ cho
        train ổn định. Đây là lỗi kinh điển và không hề báo gì, chỉ khiến loss xấu hơn.
        """
        decay = [p for p in self.parameters() if p.dim() >= 2 and p.requires_grad]
        no_decay = [p for p in self.parameters() if p.dim() < 2 and p.requires_grad]
        return [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
