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
EVAL_DIR = DATA_DIR / "eval"
FINAL_EVAL_PATH = EVAL_DIR / "final.jsonl"
FINAL_HUMAN_EVAL_PATH = EVAL_DIR / "final_human.jsonl"
FINAL_HUMAN_META_PATH = EVAL_DIR / "final_human.meta.json"
GEN_DIAGNOSTIC_PATH = EVAL_DIR / "generation_diagnostic.json"
GEN_FINAL_PATH = EVAL_DIR / "generation_final.json"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
TOKENIZER_DIR = ARTIFACT_DIR / "tokenizer"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
TOKENIZER_PATH = TOKENIZER_DIR / "luna_zero_bpe.json"
RELEASE_DIR = ARTIFACT_DIR / "release"
STAGE4_REPORT_PATH = ARTIFACT_DIR / "eval" / "stage4_report.json"
STAGE4_REPORT_MD_PATH = ARTIFACT_DIR / "eval" / "STAGE4_REPORT.md"


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


# --- Dữ liệu đã đóng gói ---
@dataclass(frozen=True)
class DataConfig:
    """Tham số cho bước khử trùng lặp và đóng gói token."""

    # 0,1% của 2,2 tỷ token ~ 2,2 triệu token cho tập val. Thừa sức đo loss ổn định,
    # mà vẫn không cắt mất phần đáng kể của dữ liệu train.
    val_ratio: float = 0.001
    # uint16 vì vocab 32.000 < 65.536. Dùng uint32 sẽ làm file .bin to gấp đôi
    # (4,4GB -> 8,8GB) mà không mang thêm thông tin nào.
    token_dtype: str = "uint16"
    # Trần lý thuyết của uint16. `pack` phải kiểm chứ không được tin.
    max_token_id: int = 65_535
    # Ghi ra đĩa mỗi 50.000 document: đủ để I/O không vụn, đủ nhỏ để RAM không phình.
    flush_every_docs: int = 50_000


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
    # Lưu mỗi 100 bước ~ 18 phút công sức. Máy thường chỉ chạy được 1-2 tiếng mỗi
    # phiên, nên mất 36 phút vì cúp điện là mất một phần đáng kể của cả phiên.
    # Mỗi lần ghi 1,3GB mất vài giây trên SSD -> phụ trội dưới 0,5% thời gian train.
    save_every_steps: int = 100
    # ĐO THẬT trên RTX 3060 12GB: 23.900 token/s ổn định suốt hàng chục bước liên tiếp
    # (bf16 + flash attention, đã đồng bộ CUDA trước khi bấm giờ). Nhịp tụt xuống
    # 20-21k chỉ xuất hiện ngay sau mỗi lần lưu checkpoint — đó là eval + ghi đĩa lọt
    # vào khoảng đo, khoảng 1,5% phụ trội.
    #
    # Đặt 22.000 làm biên. Hai con số trước đều là bài học: 6.000 là phỏng đoán thuần
    # (sai 4 lần), và cú tụt xuống 0,5k từng bị mình quy cho ổ cứng cơ — thật ra là
    # MÁY VÀO CHẾ ĐỘ NGỦ. Nhớ tắt sleep bằng powercfg trước mỗi phiên train dài.
    tokens_per_second_3060: int = 22_000


# --- Eval ------------------------------------------------------------------
@dataclass(frozen=True)
class EvalConfig:
    """Mặc định cho final eval độc lập với tập validation dùng chọn checkpoint."""

    batch_size: int = 4
    smoke_docs: int = 4
    smoke_windows: int = 2
    max_overlap_examples: int = 5


# --- Sinh văn bản ----------------------------------------------------------
@dataclass(frozen=True)
class SinhEvalConfig:
    """Cấu hình benchmark sinh; final khóa trên profile mặc định đã chọn."""

    seeds: tuple[int, ...] = (17, 29, 43)
    final_checkpoint_step: int = 33_569
    final_checkpoint_fingerprint: str = "1d6444caa222957a0fc266a9c511fe8e"
    final_profile_name: str = "mac_dinh"
    # Fingerprint semantic của generation_final.json. Đổi suite phải là hành động
    # chủ ý và cập nhật khóa này; sửa prompt âm thầm sẽ bị từ chối trước khi sinh.
    final_suite_fingerprint: str = "fe9ce37b768210b25053f96a4fee06c1"
    smoke_prompts: int = 2
    smoke_seeds: int = 1
    smoke_tokens: int = 24
    legacy_temperature: float = 0.8
    legacy_top_k: int = 50
    legacy_top_p: float | None = None
    legacy_phat_lap: float = 1.0


@dataclass(frozen=True)
class ReleaseConfig:
    """Tên artifact và schema của gói inference/public release."""

    schema_version: int = 1
    package_dir_name: str = "luna-zero-110m"
    weights_name: str = "model.safetensors"
    model_config_name: str = "config.json"
    generation_config_name: str = "generation_config.json"
    model_card_name: str = "README.md"
    tokenizer_name: str = "luna_zero_bpe.json"
    tokenizer_meta_name: str = "luna_zero_bpe.meta.json"
    stage4_report_name: str = "stage4_report.json"
    stage4_report_md_name: str = "STAGE4_REPORT.md"
    final_generation_name: str = "generation_final.json"


@dataclass(frozen=True)
class SinhConfig:
    """Tham số lấy mẫu mặc định.

    Trước đây chúng chỉ tồn tại dưới dạng default của argparse trong sinh_thu.py. Khi
    giao diện chat cần đúng bộ tham số ấy thì đã có nguy cơ thành hai bản lệch nhau —
    đúng lỗi số 6. Khai một chỗ, cả CLI lẫn web cùng import.
    """

    so_token: int = 120
    temperature: float = 0.9
    top_k: int | None = None
    # nucleus; 0 để tắt
    top_p: float = 0.92
    # 1.0 = tắt. 1.15 đè lặp quá tay (distinct-2 vượt mức người viết).
    phat_lap: float = 1.05


def chon_thiet_bi(uu_tien: str | None = None) -> str:
    """Chọn thiết bị tính toán. ĐÂY LÀ CHỖ DUY NHẤT được phép nhắc tới "cuda".

    Train thì bắt buộc GPU (CPU chậm hơn 50-100 lần), nhưng model phải CHẠY được trên
    máy không GPU — đó là điểm Luna Zero hơn hẳn Luna cũ. Viết cứng .cuda() rải rác
    trong code là cách chắc chắn nhất để đánh mất điều đó mà không ai nhận ra.
    """
    if uu_tien:
        return uu_tien
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


TOKENIZER = TokenizerConfig()
DATA = DataConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
EVAL = EvalConfig()
SINH_EVAL = SinhEvalConfig()
RELEASE = ReleaseConfig()
SINH = SinhConfig()
