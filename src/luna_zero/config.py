"""Nguồn hằng số DUY NHẤT của Luna Zero.

Bài học từ Luna cũ: system prompt từng tồn tại 3 bản ở 3 script rồi lệch nhau lúc nào
không hay. Nên mọi con số cấu hình chỉ được khai ở đây; script và test import vào,
không được viết lại literal. `tests/test_config_single_source.py` tự động chặn việc đó.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Đường dẫn -------------------------------------------------------------
# parents[2] = <repo>/src/luna_zero/config.py -> <repo>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
TOKENIZER_DIR = ARTIFACT_DIR / "tokenizer"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
TOKENIZER_PATH = TOKENIZER_DIR / "luna_zero_bpe.json"


# --- Tokenizer -------------------------------------------------------------
# Token đặc biệt: id cố định 0..2. UNK cố ý KHÔNG tồn tại — BPE mức byte phủ được
# mọi chuỗi UTF-8, nên không bao giờ cần UNK; có UNK chỉ tạo chỗ để mất dữ liệu âm thầm.
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 32_000
    # Số byte văn bản dùng để train tokenizer. 500MB đã bão hoà với vocab 32k;
    # đưa cả 6GB vào chỉ tốn thời gian chứ merge table gần như không đổi.
    train_bytes: int = 500 * 1024 * 1024
    min_frequency: int = 2
    # Ngưỡng nén tối thiểu (ký tự/token) mà test bắt buộc phải đạt.
    # Đặt thấp hơn mục tiêu 2,5-3,0 một chút để test không đỏ vì nhiễu lấy mẫu.
    min_chars_per_token: float = 2.4


# --- Model -----------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """GPT-2 small cỡ tiếng Việt. Đã chốt, không tính lại."""

    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    block_size: int = 1024
    vocab_size: int = 32_000
    dropout: float = 0.0
    bias: bool = False


# --- Train -----------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    target_tokens: int = 2_200_000_000
    micro_batch_size: int = 8
    grad_accum_steps: int = 8
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 2_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # Bài học từ Luna cũ: save_strategy="epoch" không giới hạn từng ngốn 1,4GB đĩa.
    # Luôn có trần số checkpoint giữ lại.
    max_checkpoints_keep: int = 3
    # Lưu mỗi 200 bước ~ 36 phút công sức. Mất tối đa chừng đó khi cúp điện.
    # Xuống 50 thì I/O (1,3GB/lần ghi) bắt đầu ăn vào thời gian train.
    save_every_steps: int = 200
    # Tốc độ đo thực tế trên RTX 3060 12GB, dùng để ước lượng thời gian chạy.
    tokens_per_second_3060: int = 6_000


TOKENIZER = TokenizerConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
