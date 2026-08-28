"""Tỷ lệ nén không được tệ hơn ngưỡng."""

from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.slow
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


def test_fixture_khong_lan_vao_corpus_train(sample_texts) -> None:
    """Chặn ô nhiễm eval ngay từ đầu.

    Bài học Luna cũ: 8/10 câu eval nằm nguyên văn trong data train, nên eval chỉ đo
    trí nhớ và bỏ lọt một bước lùi thật. Test này đỏ nếu ai đó copy fixture vào
    data/raw/ để "cho corpus phong phú hơn".
    """
    raw = Path(config.RAW_DIR)
    if not raw.exists():
        pytest.skip("chưa có corpus")
    needles = {t[:80] for t in sample_texts}
    for path in raw.glob("*.jsonl"):
        # Quét theo dòng: shard corpus tới 256MB, đọc cả file vào RAM là tự bắn chân.
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                trung = [n for n in needles if n in line]
                assert not trung, f"{path.name} chứa câu fixture: {trung[:1]}"
