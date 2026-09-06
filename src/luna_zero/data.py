"""Đọc, chuẩn hoá và lấy mẫu corpus thô."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path

# Ký tự điều khiển trừ \t \n \r — thường là rác từ crawl, cắt sớm cho sạch.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_BLANK = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Chuẩn hoá NFC + bỏ ký tự điều khiển + gom dòng trống.

    NFC được làm Ở ĐÂY (tầng dữ liệu) chứ không phải trong tokenizer. Lý do: tiếng Việt
    có hai cách mã hoá dấu (tổ hợp NFD vs dựng sẵn NFC) cho cùng một chữ, corpus crawl
    lẫn cả hai. Nếu nhét normalizer vào tokenizer thì tokenizer không còn round-trip
    đúng nguyên văn nữa và test round-trip mất giá trị. Chuẩn hoá corpus trước, giữ
    tokenizer không mất mát.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_CHARS.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def doc_hash(text: str) -> str:
    """Vân tay document sau NFC, dùng chung cho split và kiểm contamination.

    Hash ở tầng dữ liệu để mọi phép chia/kiểm eval dùng đúng một định nghĩa. Việc
    chuẩn hoá ngay trong hàm làm NFD/NFC không thể lách qua kiểm tra trùng.
    """
    clean = normalize_text(text)
    return hashlib.blake2b(clean.encode("utf-8"), digest_size=16).hexdigest()


def iter_jsonl_texts(path: Path, field: str = "text") -> Iterator[str]:
    """Duyệt từng document trong file .jsonl. Bỏ qua dòng hỏng thay vì chết cả job."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = obj.get(field)
            if isinstance(value, str) and value:
                yield value


def iter_corpus(root: Path, max_bytes: int | None = None) -> Iterator[str]:
    """Duyệt mọi .jsonl trong `root` (đã sắp xếp để lặp lại được), dừng khi đủ byte."""
    seen = 0
    for path in sorted(root.glob("*.jsonl")):
        for text in iter_jsonl_texts(path):
            yield text
            if max_bytes is not None:
                seen += len(text.encode("utf-8"))
                if seen >= max_bytes:
                    return


def write_jsonl(path: Path, texts: Iterator[str]) -> tuple[int, int]:
    """Ghi corpus ra .jsonl. Trả về (số document, số byte văn bản)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_docs = n_bytes = 0
    with path.open("w", encoding="utf-8") as f:
        for text in texts:
            clean = normalize_text(text)
            if not clean:
                continue
            f.write(json.dumps({"text": clean}, ensure_ascii=False) + "\n")
            n_docs += 1
            n_bytes += len(clean.encode("utf-8"))
    return n_docs, n_bytes
