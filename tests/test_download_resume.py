"""Chạy tiếp lần tải bị ngắt, không ghi trùng.

Ctrl+C giữa chừng khi tải 8,5GB là chuyện sẽ xảy ra, không phải có thể xảy ra.
Nếu lần chạy sau đọc lại từ đầu stream thì corpus có 890.000 document trùng nguyên văn —
và trùng lặp trong dữ liệu pretraining làm model học thuộc thay vì học khái quát,
đúng họ hàng với bài học "eval trùng dữ liệu train" của Luna cũ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from download_corpus import dem_da_co  # noqa: E402


def _viet_shard(path: Path, texts: list[str]) -> None:
    path.write_text(
        "".join(json.dumps({"text": t}, ensure_ascii=False) + "\n" for t in texts),
        encoding="utf-8",
    )


def test_thu_muc_rong_thi_bat_dau_tu_dau(tmp_path: Path) -> None:
    assert dem_da_co(tmp_path, "culturax") == (0, 0, 0)


def test_dem_dung_so_doc_va_byte(tmp_path: Path) -> None:
    _viet_shard(tmp_path / "culturax_vi_0000.jsonl", ["Tiếng Việt", "có dấu"])
    _viet_shard(tmp_path / "culturax_vi_0001.jsonl", ["thêm một"])
    n_docs, n_bytes, shard_ke = dem_da_co(tmp_path, "culturax")
    assert n_docs == 3
    # Đếm theo byte UTF-8 của văn bản, không phải kích thước file (file có thêm JSON).
    assert n_bytes == sum(len(t.encode("utf-8")) for t in ("Tiếng Việt", "có dấu", "thêm một"))
    assert shard_ke == 2


def test_khong_lan_sang_nguon_khac(tmp_path: Path) -> None:
    """Wikipedia và CulturaX dùng chung thư mục thì không được đếm nhầm của nhau."""
    _viet_shard(tmp_path / "culturax_vi_0000.jsonl", ["a", "b"])
    _viet_shard(tmp_path / "wikipedia_vi_0000.jsonl", ["c", "d", "e"])
    assert dem_da_co(tmp_path, "culturax")[0] == 2
    assert dem_da_co(tmp_path, "wikipedia")[0] == 3


def test_bo_qua_dong_hong_khong_dem_thanh_doc(tmp_path: Path) -> None:
    """Shard cuối bị Ctrl+C cắt dở có thể để lại một dòng JSON không trọn vẹn."""
    path = tmp_path / "culturax_vi_0000.jsonl"
    path.write_text(
        json.dumps({"text": "tron ven"}) + "\n" + '{"text": "bi cat dor',
        encoding="utf-8",
    )
    assert dem_da_co(tmp_path, "culturax")[0] == 1


def test_shard_ke_tiep_khong_de_len_shard_cu(tmp_path: Path) -> None:
    """Phép đo chiều ngược: lần chạy tiếp KHÔNG được phép ghi đè file đã có."""
    for i in (0, 1, 2):
        _viet_shard(tmp_path / f"culturax_vi_{i:04d}.jsonl", ["x"])
    _, _, shard_ke = dem_da_co(tmp_path, "culturax")
    assert not (tmp_path / f"culturax_vi_{shard_ke:04d}.jsonl").exists()


@pytest.mark.parametrize("thieu", ["", "   \n\n"])
def test_dong_trong_khong_tinh(tmp_path: Path, thieu: str) -> None:
    (tmp_path / "culturax_vi_0000.jsonl").write_text(
        json.dumps({"text": "mot"}) + "\n" + thieu, encoding="utf-8"
    )
    assert dem_da_co(tmp_path, "culturax")[0] == 1
