"""Khử trùng lặp, chia train/val, và đóng gói token — ba chỗ hỏng im lặng."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from luna_zero import config
from luna_zero.pack import (
    PackStats,
    _BinWriter,
    chon_split,
    doc_hash,
    load_split,
    loc_trung_lap,
    pack_documents,
)


# --- vân tay document ------------------------------------------------------
def test_hash_on_dinh_va_khac_nhau_cho_van_ban_khac() -> None:
    assert doc_hash("xin chào") == doc_hash("xin chào")
    assert doc_hash("xin chào") != doc_hash("xin chao")


def test_hash_bo_qua_khac_biet_ma_hoa_dau() -> None:
    """NFD và NFC của cùng một chữ phải cho cùng vân tay sau khi chuẩn hoá."""
    import unicodedata

    from luna_zero.data import normalize_text

    nfd = unicodedata.normalize("NFD", "Tiếng Việt")
    assert doc_hash(normalize_text(nfd)) == doc_hash("Tiếng Việt")


# --- khử trùng lặp ---------------------------------------------------------
def test_bo_ban_trung_giu_ban_dau_tien() -> None:
    docs = ["một", "hai", "một", "ba", "hai"]
    giu = [t for t, _ in loc_trung_lap(docs)]
    assert giu == ["một", "hai", "ba"]


def test_khong_bo_nham_van_ban_chi_gan_giong() -> None:
    """Chỉ khử trùng NGUYÊN VĂN. Hai câu khác một chữ là hai document thật."""
    docs = ["Hà Nội mùa thu", "Hà Nội mùa thu."]
    assert len([t for t, _ in loc_trung_lap(docs)]) == 2


# --- chia train/val --------------------------------------------------------
def test_split_tat_dinh_theo_noi_dung() -> None:
    """Cùng một document luôn rơi vào cùng một phía, bất kể thứ tự hay lần chạy."""
    text = "Một document bất kỳ để kiểm tra tính tất định."
    assert chon_split(text) == chon_split(text)


def test_val_va_train_khong_bao_gio_giao_nhau() -> None:
    """Bài học Luna cũ: eval trùng train thì phép đo chỉ đo trí nhớ.

    Phép đo chiều ngược: không quan tâm tỷ lệ chia có đẹp không, chỉ đòi một điều —
    KHÔNG document nào xuất hiện ở cả hai phía.
    """
    docs = [f"Document số {i} với nội dung riêng biệt." for i in range(3_000)]
    train = {t for t, s in loc_trung_lap(docs) if s == "train"}
    val = {t for t, s in loc_trung_lap(docs) if s == "val"}
    assert not (train & val)
    assert len(train) + len(val) == len(docs)


def test_ty_le_val_xap_xi_dung() -> None:
    docs = [f"doc {i}" for i in range(20_000)]
    n_val = sum(1 for _, s in loc_trung_lap(docs) if s == "val")
    # val_ratio=0.05 trên 20k mẫu: sai số lấy mẫu vài phần trăm là bình thường.
    n_val_5 = sum(1 for t in docs if chon_split(t, val_ratio=0.05) == "val")
    assert 800 <= n_val_5 <= 1_200, n_val_5
    assert n_val < len(docs) * 0.01


# --- ghi nhị phân ----------------------------------------------------------
def test_token_vuot_tran_uint16_bi_chan_chu_khong_am_tham_sai(tmp_path: Path) -> None:
    """Phép đo chiều ngược quan trọng nhất của cả module.

    numpy ép int64 xuống uint16 bằng phép lấy dư và KHÔNG báo gì. Một token id 70.000
    sẽ lặng lẽ thành 4.464 — dữ liệu hỏng nhưng loss vẫn giảm đẹp, và không có cách nào
    phát hiện sau này ngoài việc đọc lại từng token. Phải chặn tại chỗ ghi.
    """
    w = _BinWriter(tmp_path / "x.bin")
    w.write([1, 2, config.DATA.max_token_id + 1])
    with pytest.raises(ValueError, match="ngoài khoảng uint16"):
        w.flush()


def test_token_am_cung_bi_chan(tmp_path: Path) -> None:
    w = _BinWriter(tmp_path / "x.bin")
    w.write([1, -1])
    with pytest.raises(ValueError):
        w.flush()


def test_vocab_hien_tai_nam_gon_trong_uint16() -> None:
    assert config.TOKENIZER.vocab_size <= config.DATA.max_token_id + 1


# --- đóng gói đầu-cuối -----------------------------------------------------
def test_pack_roi_doc_lai_ra_dung_token(tiny_tokenizer, tmp_path: Path) -> None:
    docs = ["Hà Nội mùa thu.", "Sài Gòn mùa mưa."]
    stats = pack_documents(tiny_tokenizer, docs, tmp_path, val_ratio=0.0)
    assert stats.n_docs_vao == 2
    assert stats.n_docs_trung == 0

    mong_doi = [i for d in docs for i in tiny_tokenizer.encode_document(d)]
    assert load_split(tmp_path, "train").tolist() == mong_doi
    assert load_split(tmp_path, "val").size == 0


def test_pack_dem_dung_ban_trung(tiny_tokenizer, tmp_path: Path) -> None:
    stats = pack_documents(tiny_tokenizer, ["a", "b", "a", "a"], tmp_path, val_ratio=0.0)
    assert (stats.n_docs_vao, stats.n_docs_trung, stats.n_docs_giu) == (4, 2, 2)


def test_meta_json_khop_voi_file_that(tiny_tokenizer, tmp_path: Path) -> None:
    """Không tin bộ đếm trong bộ nhớ: đối chiếu với kích thước file thật trên đĩa."""
    docs = [f"Câu số {i} trong corpus thử." for i in range(50)]
    pack_documents(tiny_tokenizer, docs, tmp_path, val_ratio=0.2)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_train_tokens"] == load_split(tmp_path, "train").size
    assert meta["n_val_tokens"] == load_split(tmp_path, "val").size
    assert meta["n_tokens"] == meta["n_train_tokens"] + meta["n_val_tokens"]
    assert meta["vocab_size"] == tiny_tokenizer.vocab_size


def test_file_bin_dung_kieu_uint16(tiny_tokenizer, tmp_path: Path) -> None:
    pack_documents(tiny_tokenizer, ["một câu"], tmp_path, val_ratio=0.0)
    arr = load_split(tmp_path, "train")
    assert arr.dtype == np.uint16
    assert (tmp_path / "train.bin").stat().st_size == arr.size * 2


def test_corpus_rong_khong_no(tiny_tokenizer, tmp_path: Path) -> None:
    stats = pack_documents(tiny_tokenizer, [], tmp_path)
    assert stats.n_tokens == 0
    assert load_split(tmp_path, "train").size == 0


def test_pack_stats_tinh_dung() -> None:
    s = PackStats(n_docs_vao=10, n_docs_trung=3, n_train_tokens=100, n_val_tokens=5)
    assert s.n_docs_giu == 7
    assert s.n_tokens == 105
