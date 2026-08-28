from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from luna_zero.data import iter_jsonl_texts  # noqa: E402
from luna_zero.tokenizer import LunaTokenizer, train_tokenizer  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "vi_sample.jsonl"


@pytest.fixture(scope="session")
def sample_texts() -> list[str]:
    return list(iter_jsonl_texts(FIXTURE))


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory, sample_texts) -> LunaTokenizer:
    """Tokenizer tí hon train trên fixture 8KB — chạy dưới một giây.

    Bài học Luna cũ: code nặng phải có bản chạy được trên dữ liệu tí hon, nếu không
    thì lỗi pipeline chỉ lộ ra sau khi đã đốt hàng chục phút CPU.
    """
    out = tmp_path_factory.mktemp("tok") / "tiny.json"
    train_tokenizer(sample_texts, output_path=out, vocab_size=600, min_frequency=1)
    return LunaTokenizer.load(out)
