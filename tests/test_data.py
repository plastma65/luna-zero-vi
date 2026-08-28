"""Chuẩn hoá văn bản: NFC, bỏ ký tự điều khiển, gom dòng trống."""

from __future__ import annotations

import unicodedata

from luna_zero.data import iter_jsonl_texts, normalize_text, write_jsonl


def test_nfc_gop_dau_tach_roi() -> None:
    # "ế" viết kiểu tổ hợp (NFD) phải thành một ký tự dựng sẵn (NFC).
    nfd = unicodedata.normalize("NFD", "Tiếng Việt")
    assert len(nfd) > len("Tiếng Việt")
    assert normalize_text(nfd) == "Tiếng Việt"


def test_bo_ky_tu_dieu_khien_giu_tab_va_xuong_dong() -> None:
    assert normalize_text("a\x00b\x07c") == "abc"
    assert normalize_text("a\tb\nc") == "a\tb\nc"


def test_gom_dong_trong_thua() -> None:
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"


def test_jsonl_round_trip_giu_nguyen_dau(tmp_path) -> None:
    texts = ["Chào buổi sáng, Hà Nội.", "Nghiêng nghiêng."]
    path = tmp_path / "x.jsonl"
    n_docs, n_bytes = write_jsonl(path, iter(texts))
    assert n_docs == 2
    assert n_bytes > 0
    assert list(iter_jsonl_texts(path)) == texts


def test_bo_qua_dong_json_hong(tmp_path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"text": "ok"}\nkhong-phai-json\n{"text": "hai"}\n', encoding="utf-8")
    assert list(iter_jsonl_texts(path)) == ["ok", "hai"]
