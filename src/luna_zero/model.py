"""Khung mô hình GPT của Luna Zero.

Chặng 1 chưa dựng mạng (chưa cài torch). Phần có ở đây là phép đếm tham số — nó
kiểm chứng được con số 110M đã chốt mà không cần GPU, và test bám vào nó để nếu ai
sửa ModelConfig làm model phình ra ngoài tầm VRAM thì CI đỏ ngay.
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


# Ước lượng thô bộ nhớ kích hoạt: mỗi phần tử ẩn kéo theo khoảng 20 tensor trung gian
# còn sống trong một lớp transformer (qkv, softmax, gelu, residual...). Con số này là
# kinh nghiệm đo được chứ không phải công thức chính xác, nhưng nó phản ứng ĐÚNG CHIỀU
# với block_size / batch / d_model / n_layer — đủ để chặn một thay đổi config làm tràn VRAM.
ACTIVATION_FACTOR = 20
BYTES_FP16 = 2
# AdamW hỗn hợp: trọng số fp32 + gradient fp32 + hai trạng thái Adam = 16 byte/tham số.
BYTES_PER_PARAM_ADAMW = 16


def estimate_vram_gb(model: ModelConfig = MODEL, train: TrainConfig = TRAIN) -> float:
    """Ước lượng VRAM lúc train, đơn vị GB. Giả định dùng attention hợp nhất (SDPA/flash),
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
