"""Khử trùng lặp và đóng gói corpus thành mảng token nhị phân.

Kết quả là hai file phẳng `train.bin` / `val.bin` chứa token uint16 nối đuôi nhau,
kèm `meta.json`. Vòng lặp train sẽ memmap chúng và cắt cửa sổ block_size ở vị trí
ngẫu nhiên — không cần nạp cả 4,4GB vào RAM.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from luna_zero import config
from luna_zero.data import doc_hash
from luna_zero.tokenizer import LunaTokenizer


def tokenizer_fingerprint(path: Path) -> str:
    """Vân tay của chính FILE tokenizer, không phải của các tham số nó khai báo.

    vocab_size không đủ để nhận dạng: hai tokenizer train trên hai corpus khác nhau vẫn
    cùng vocab 32.000 nhưng bảng merge khác hẳn, nên cùng một id trỏ vào hai token khác
    nhau. Băm nguyên file là cách duy nhất phân biệt được.
    """
    return hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()


def kiem_tokenizer_khop(out_dir: Path, tokenizer_path: Path) -> None:
    """Chặn việc train trên .bin sinh bởi tokenizer KHÁC tokenizer đang dùng.

    Đây là hỏng im lặng đắt nhất của cả dự án: id trong train.bin trỏ vào bảng merge cũ,
    loss vẫn giảm bình thường, train vẫn chạy đủ 4 ngày, và chỉ khi sinh văn bản mới lộ
    ra toàn chữ rác. Ném lỗi ngay ở bước khởi động thay vì để phát hiện sau 4 ngày.
    """
    meta_file = out_dir / "meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Thiếu {meta_file}. Chạy scripts/dong_goi.py trước.")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    da_ghi = meta.get("tokenizer_fingerprint")
    if da_ghi is None:
        raise ValueError(
            f"{meta_file} không ghi vân tay tokenizer (đóng gói bằng bản cũ). "
            "Chạy lại scripts/dong_goi.py để sinh meta.json đầy đủ."
        )
    hien_tai = tokenizer_fingerprint(tokenizer_path)
    if da_ghi != hien_tai:
        raise ValueError(
            "TOKENIZER KHÔNG KHỚP DỮ LIỆU ĐÃ ĐÓNG GÓI.\n"
            f"  {out_dir.name}/meta.json : {da_ghi}\n"
            f"  {tokenizer_path.name}    : {hien_tai}\n"
            "Token id trong .bin trỏ vào bảng merge khác. Train tiếp sẽ cho model "
            "sinh ra chữ rác mà loss vẫn đẹp. Chạy lại scripts/dong_goi.py."
        )


def kiem_tokenizer_checkpoint(
    blob: dict[str, object],
    tokenizer_path: Path,
    data_dir: Path | None = None,
) -> str:
    """Xác minh tokenizer khi nạp checkpoint; vocab_size một mình không đủ.

    Checkpoint mới mang fingerprint trực tiếp. Với checkpoint cũ đã train trước khi
    trường này tồn tại, chỉ cho phép fallback qua `data/processed/meta.json` đã được
    đóng gói bằng đúng tokenizer. Không có cả hai bằng chứng thì từ chối suy đoán.
    """
    hien_tai = tokenizer_fingerprint(tokenizer_path)
    da_ghi = blob.get("tokenizer_fingerprint")
    if isinstance(da_ghi, str):
        if da_ghi != hien_tai:
            raise ValueError(
                "TOKENIZER KHÔNG KHỚP CHECKPOINT.\n"
                f"  checkpoint : {da_ghi}\n"
                f"  tokenizer  : {hien_tai}"
            )
        return "checkpoint"

    data_dir = Path(data_dir) if data_dir is not None else config.PROCESSED_DIR
    if (data_dir / "meta.json").exists():
        kiem_tokenizer_khop(data_dir, tokenizer_path)
        return "data_meta"
    raise ValueError(
        "Checkpoint cũ không có vân tay tokenizer và cũng không có data/processed/meta.json "
        "để đối chiếu. Không thể chứng minh tokenizer hiện tại là bản đã dùng để train."
    )


def chon_split(text: str, val_ratio: float = config.DATA.val_ratio) -> str:
    """Quyết định document thuộc train hay val, TẤT ĐỊNH theo nội dung.

    Bài học Luna cũ: eval trùng dữ liệu train thì phép đo chỉ đo trí nhớ. Chia theo
    hash nội dung (không phải theo thứ tự hay số ngẫu nhiên) bảo đảm một document
    luôn rơi vào cùng một phía, kể cả khi corpus được tải lại theo thứ tự khác hay
    có thêm shard mới. Nhờ vậy tập val không bao giờ lẫn sang train giữa các lần chạy.
    """
    # 8 hex đầu -> số nguyên 32 bit, chia đều trong [0, 1).
    diem = int(doc_hash(text)[:8], 16) / 0x1_0000_0000
    return "val" if diem < val_ratio else "train"


def loc_trung_lap(docs: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Sinh (text, split) cho các document chưa từng thấy. Giữ bản xuất hiện đầu tiên."""
    da_thay: set[str] = set()
    for text in docs:
        h = doc_hash(text)
        if h in da_thay:
            continue
        da_thay.add(h)
        yield text, chon_split(text)


@dataclass
class PackStats:
    n_docs_vao: int = 0
    n_docs_trung: int = 0
    n_train_tokens: int = 0
    n_val_tokens: int = 0

    @property
    def n_docs_giu(self) -> int:
        return self.n_docs_vao - self.n_docs_trung

    @property
    def n_tokens(self) -> int:
        return self.n_train_tokens + self.n_val_tokens

    def to_dict(self) -> dict[str, int]:
        d = asdict(self)
        d.update(n_docs_giu=self.n_docs_giu, n_tokens=self.n_tokens)
        return d


class _BinWriter:
    """Ghi token ra file nhị phân theo lô, có kiểm tràn uint16."""

    def __init__(self, path: Path, dtype: str = config.DATA.token_dtype) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("wb")
        self._dtype = np.dtype(dtype)
        self._buf: list[int] = []
        self.n_tokens = 0

    def write(self, ids: list[int]) -> None:
        self._buf.extend(ids)
        self.n_tokens += len(ids)

    def flush(self) -> None:
        if not self._buf:
            return
        arr = np.asarray(self._buf, dtype=np.int64)
        # Kiểm TRƯỚC khi ép kiểu. numpy ép xuống uint16 bằng cách lấy dư 65536 và
        # KHÔNG báo lỗi, nên một token id vượt trần sẽ âm thầm biến thành token khác —
        # dữ liệu hỏng mà loss vẫn giảm bình thường. Đây là lỗi phải chặn, không phải cảnh báo.
        if arr.size and (arr.max() > config.DATA.max_token_id or arr.min() < 0):
            raise ValueError(
                f"token id ngoài khoảng uint16 (max={arr.max()}, min={arr.min()}). "
                f"vocab_size={config.TOKENIZER.vocab_size} có vượt {config.DATA.max_token_id}?"
            )
        self._fh.write(arr.astype(self._dtype).tobytes())
        self._buf.clear()

    def close(self) -> None:
        self.flush()
        self._fh.close()


def pack_documents(
    tokenizer: LunaTokenizer,
    docs: Iterable[str],
    out_dir: Path,
    tokenizer_path: Path | None = None,
    val_ratio: float = config.DATA.val_ratio,
    tien_do_moi: int = config.DATA.flush_every_docs,
    im_lang: bool = False,
) -> PackStats:
    """Khử trùng lặp, token hoá, ghi ra train.bin / val.bin / meta.json.

    In tiến trình theo mặc định. Công việc này chạy nửa tiếng; một tiến trình im lặng
    lâu đến vậy không phân biệt được với treo, và người chạy sẽ giết nhầm nó.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        "train": _BinWriter(out_dir / "train.bin"),
        "val": _BinWriter(out_dir / "val.bin"),
    }
    stats = PackStats()
    da_thay: set[str] = set()
    t0 = time.perf_counter()
    try:
        for text in docs:
            stats.n_docs_vao += 1
            h = doc_hash(text)
            if h in da_thay:
                stats.n_docs_trung += 1
                continue
            da_thay.add(h)
            split = chon_split(text, val_ratio)
            writers[split].write(tokenizer.encode_document(text))
            if stats.n_docs_vao % tien_do_moi == 0:
                for w in writers.values():
                    w.flush()
                if not im_lang:
                    tok_da_ghi = writers["train"].n_tokens + writers["val"].n_tokens
                    giay = time.perf_counter() - t0
                    print(
                        f"   {stats.n_docs_vao:>9,} doc | {tok_da_ghi / 1e6:>7.1f}M token"
                        f" | trùng {stats.n_docs_trung:>7,}"
                        f" | {tok_da_ghi / giay / 1e3:>5.0f}k token/s",
                        flush=True,
                    )
    finally:
        for w in writers.values():
            w.close()

    stats.n_train_tokens = writers["train"].n_tokens
    stats.n_val_tokens = writers["val"].n_tokens
    meta = {
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_fingerprint": (
            tokenizer_fingerprint(tokenizer_path) if tokenizer_path else None
        ),
        "dtype": config.DATA.token_dtype,
        "val_ratio": val_ratio,
        **stats.to_dict(),
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


def load_split(out_dir: Path, split: str) -> np.ndarray:
    """Memmap một split. Chỉ đọc, không nạp cả file vào RAM.

    np.memmap không mở được file 0 byte, mà file rỗng là trạng thái hợp lệ (val_ratio=0,
    hoặc corpus quá nhỏ nên không document nào rơi vào val). Trả mảng rỗng thay vì nổ.
    """
    path = out_dir / f"{split}.bin"
    if not path.exists() or path.stat().st_size == 0:
        return np.empty(0, dtype=config.DATA.token_dtype)
    return np.memmap(path, dtype=config.DATA.token_dtype, mode="r")
