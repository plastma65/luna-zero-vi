"""Đếm tham số và ước lượng VRAM — số học thuần, KHÔNG cần torch.

Tách khỏi model.py có lý do: đây là phép tính quyết định model có vừa card hay không và
train mất bao lâu, nên nó phải chạy được ở mọi nơi — máy chưa cài torch, CI, hay lúc chỉ
muốn thử một cấu hình mới. Bắt nó kéo theo gói 2,5GB là vô lý.
"""

from __future__ import annotations

from luna_zero.config import MODEL, TRAIN, ModelConfig, TrainConfig


def estimate_num_params(cfg: ModelConfig = MODEL, tie_embeddings: bool = True) -> int:
    """Đếm tham số của GPT decoder-only theo công thức, không cần khởi tạo mạng."""
    d = cfg.d_model
    bias = 1 if cfg.bias else 0

    token_emb = cfg.vocab_size * d
    pos_emb = cfg.block_size * d

    # Mỗi lớp: attention qkv + proj, MLP 4d, hai LayerNorm.
    attn = 4 * d * d + bias * 4 * d
    mlp = 8 * d * d + bias * 5 * d
    norms = 2 * d * (1 + bias)
    per_layer = attn + mlp + norms

    total = token_emb + pos_emb + cfg.n_layer * per_layer + d * (1 + bias)
    if not tie_embeddings:
        total += cfg.vocab_size * d
    return total


def format_params(n: int) -> str:
    return f"{n / 1e6:.1f}M"


# HIỆU CHỈNH THEO ĐO THẬT (RTX 3060, 2026-09-02). Bản đầu đặt 20 và cho ước lượng
# 4,46GB, trong khi torch.cuda.max_memory_allocated() báo 6,91GB — hụt 55%.
# Suy ngược từ số thật:
#   kích hoạt = 6,91GB - 1,77GB (trạng thái AdamW) = 5,14GB
#   hệ số     = 5,14GB / (8 x 1024 x 768 x 12 x 2 byte) ~ 36
# Đây là con số ĐO ĐƯỢC, không phải công thức. Đo lại nếu đổi kiến trúc.
ACTIVATION_FACTOR = 36
BYTES_FP16 = 2
# AdamW hỗn hợp: trọng số fp32 + gradient fp32 + hai trạng thái Adam = 16 byte/tham số.
BYTES_PER_PARAM_ADAMW = 16


def estimate_vram_gb(model: ModelConfig = MODEL, train: TrainConfig = TRAIN) -> float:
    """Ước lượng VRAM lúc train (GB). Giả định attention hợp nhất (SDPA/flash),
    tức KHÔNG vật chất hoá ma trận attention T x T."""
    state = estimate_num_params(model) * BYTES_PER_PARAM_ADAMW
    activations = (
        train.micro_batch_size
        * model.block_size
        * model.d_model
        * model.n_layer
        * ACTIVATION_FACTOR
        * BYTES_FP16
    )
    return (state + activations) / 1024**3
