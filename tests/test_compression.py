"""Tỷ lệ nén không được tệ hơn ngưỡng."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from conftest import FIXTURE  # noqa: E402 - pytest thêm thư mục tests vào sys.path
from luna_zero import config
from luna_zero.tokenizer import LunaTokenizer, char_level_baseline, measure_compression


def test_ham_do_nen_dung_so_hoc(tiny_tokenizer, sample_texts) -> None:
    stats = measure_compression(tiny_tokenizer, sample_texts)
    base = char_level_baseline(sample_texts)
    assert stats.n_chars == base.n_chars
    assert base.chars_per_token == pytest.approx(1.0)
    # Tokenizer 600 từ vựng trên 8KB chưa thể đạt ngưỡng 32k; chỉ đòi nó nén hơn
    # mức ký tự. Ngưỡng thật được kiểm ở test dưới, trên tokenizer thật.
    assert stats.chars_per_token > 1.5


def test_do_nen_khong_am_hay_vo_han(tiny_tokenizer) -> None:
    stats = measure_compression(tiny_tokenizer, [])
    assert stats.chars_per_token == 0.0


@pytest.mark.skipif(
    not config.TOKENIZER_PATH.exists(),
    reason="chưa train tokenizer thật (chạy scripts/train_tokenizer.py)",
)
def test_tokenizer_that_dat_nguong(sample_texts) -> None:
    tok = LunaTokenizer.load(config.TOKENIZER_PATH)
    assert tok.vocab_size == config.TOKENIZER.vocab_size
    stats = measure_compression(tok, sample_texts)
    assert stats.chars_per_token >= config.TOKENIZER.min_chars_per_token, (
        f"chỉ đạt {stats.chars_per_token:.2f} ký tự/token, "
        f"cần >= {config.TOKENIZER.min_chars_per_token}"
    )


# Quét bao nhiêu byte đầu mỗi shard ở bản test nhanh. Corpus thật lên tới hàng GB;
# quét toàn bộ mất hàng phút và biến `pytest` thành thứ không ai muốn chạy.
QUET_MOI_SHARD = 8 * 1024 * 1024


def _fixture_needles(sample_texts: list[str]) -> set[str]:
    return {t[:80] for t in sample_texts}


def test_khong_ai_chep_thang_file_fixture_vao_corpus(sample_texts) -> None:
    """Bắt kiểu ô nhiễm thực tế nhất: copy nguyên file fixture vào data/raw.

    So bằng vân tay file nên chỉ tốn một lần đọc mỗi shard, không phụ thuộc corpus to cỡ nào.
    """
    raw = Path(config.RAW_DIR)
    if not raw.exists():
        pytest.skip("chưa có corpus")
    van_tay_fixture = hashlib.blake2b(FIXTURE.read_bytes(), digest_size=16).hexdigest()
    for path in raw.glob("*.jsonl"):
        if path.stat().st_size != FIXTURE.stat().st_size:
            continue  # khác kích thước thì chắc chắn khác nội dung, khỏi đọc
        van_tay = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
        assert van_tay != van_tay_fixture, f"{path.name} là bản sao của fixture"


def test_fixture_khong_lan_vao_dau_corpus(sample_texts) -> None:
    """Quét 32MB đầu mỗi shard. Nhanh, và đủ bắt trường hợp chèn vào đầu file."""
    raw = Path(config.RAW_DIR)
    if not raw.exists():
        pytest.skip("chưa có corpus")
    needles = _fixture_needles(sample_texts)
    for path in raw.glob("*.jsonl"):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            doc = f.read(QUET_MOI_SHARD)
        trung = [n for n in needles if n in doc]
        assert not trung, f"{path.name} chứa câu fixture: {trung[:1]}"


@pytest.mark.slow
def test_fixture_khong_lan_vao_corpus_train(sample_texts) -> None:
    """Bản quét đầy đủ. Chậm theo kích thước corpus nên để sau marker `slow`.

    Chạy trước mỗi lần train thật: pytest -m slow
    """
    raw = Path(config.RAW_DIR)
    if not raw.exists():
        pytest.skip("chưa có corpus")
    needles = _fixture_needles(sample_texts)
    for path in raw.glob("*.jsonl"):
        # Quét theo dòng: shard corpus tới 256MB, đọc cả file vào RAM là tự bắn chân.
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                trung = [n for n in needles if n in line]
                assert not trung, f"{path.name} chứa câu fixture: {trung[:1]}"
