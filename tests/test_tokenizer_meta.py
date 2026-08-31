"""Metadata tokenizer: ghi lại đã train trên bao nhiêu byte, để phép đo nén không tự lừa.

Bài học Luna cũ số 3: bộ eval trùng dữ liệu train thì eval chỉ đo trí nhớ. Ở đây nó
tái xuất dưới dạng khác — tokenizer train trên 2GB nhưng phép đo chỉ bỏ qua 500MB theo
hằng số mặc định, nên 1,5GB dữ liệu train lọt vào mẫu đo. Không có gì báo lỗi; tỷ lệ
nén chỉ đơn giản là cao hơn sự thật.
"""

from __future__ import annotations

import json
from pathlib import Path

from luna_zero import config
from luna_zero.tokenizer import doc_train_bytes, meta_path, train_tokenizer


def test_meta_ghi_kem_khi_train(tmp_path: Path, sample_texts) -> None:
    out = tmp_path / "t.json"
    train_tokenizer(sample_texts, output_path=out, vocab_size=600, train_bytes=12_345)
    meta = json.loads(meta_path(out).read_text(encoding="utf-8"))
    assert meta["train_bytes"] == 12_345
    assert meta["vocab_size"] == 600


def test_doc_lai_dung_so_byte(tmp_path: Path, sample_texts) -> None:
    out = tmp_path / "t.json"
    train_tokenizer(sample_texts, output_path=out, vocab_size=600, train_bytes=2_147_483_648)
    assert doc_train_bytes(out) == 2_147_483_648


def test_khong_co_meta_thi_lui_ve_config(tmp_path: Path) -> None:
    assert doc_train_bytes(tmp_path / "khong-ton-tai.json") == config.TOKENIZER.train_bytes


def test_meta_hong_khong_lam_chet_chuong_trinh(tmp_path: Path, sample_texts) -> None:
    out = tmp_path / "t.json"
    train_tokenizer(sample_texts, output_path=out, vocab_size=600, train_bytes=999)
    meta_path(out).write_text("{ day khong phai json", encoding="utf-8")
    assert doc_train_bytes(out) == config.TOKENIZER.train_bytes


def test_meta_thieu_khoa_van_chay(tmp_path: Path, sample_texts) -> None:
    out = tmp_path / "t.json"
    train_tokenizer(sample_texts, output_path=out, vocab_size=600, train_bytes=999)
    meta_path(out).write_text('{"vocab_size": 600}', encoding="utf-8")
    assert doc_train_bytes(out) == config.TOKENIZER.train_bytes


def test_meta_nam_canh_tokenizer_khong_de_len_no(tmp_path: Path, sample_texts) -> None:
    """Phép đo chiều ngược: file metadata KHÔNG được ghi đè chính file tokenizer."""
    out = tmp_path / "t.json"
    train_tokenizer(sample_texts, output_path=out, vocab_size=600, train_bytes=1)
    assert meta_path(out) != out
    assert out.exists() and meta_path(out).exists()
    assert json.loads(out.read_text(encoding="utf-8")).get("model") is not None
