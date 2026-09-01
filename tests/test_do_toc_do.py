"""Phép đo tốc độ train — số học thuần, không cần torch.

Đặt riêng khỏi test_model.py vì file đó bị bỏ qua khi máy chưa cài torch, mà chính
những test này lại bắt được lỗi thật.
"""

from __future__ import annotations

from pathlib import Path

import pytest


# --- phép đo tốc độ ---------------------------------------------------------
def test_cong_thuc_tocdo_dem_buoc_that_chu_khong_gia_dinh() -> None:
    """Chạy tiếp từ checkpoint làm khoảng in đầu tiên ngắn hơn log_moi.

    Lỗi đã xảy ra thật: chạy tiếp từ bước 27, in ở bước 30 (3 bước), nhưng công thức
    chia cho log_moi=10 -> báo 71,5k tok/s trong khi thực tế 22k. Gấp hơn ba lần, và
    không có gì báo sai. Đây lại là họ lỗi "phép đo tự lừa".
    """
    tokens_moi_buoc, giay = 65_536, 9.0

    def tps(so_buoc: int) -> float:
        return tokens_moi_buoc * so_buoc / giay

    # Ba bước thật trong 9 giây phải cho ~21,8k, không phải 72,8k.
    assert tps(3) == pytest.approx(21_845, rel=0.01)
    assert tps(10) == pytest.approx(72_818, rel=0.01)
    assert tps(10) > tps(3) * 3


def test_train_py_dem_buoc_tu_lan_in() -> None:
    """Khoá lại chính phép sửa: công thức phải dùng biến đếm, không phải hằng log_moi."""
    src = (Path(__file__).resolve().parents[1] / "src" / "luna_zero" / "train.py").read_text(
        encoding="utf-8"
    )
    assert "tps = plan.tokens_per_step * buoc_tu_lan_in / giay" in src
    assert "* log_moi / giay" not in src, "vẫn còn giả định đủ log_moi bước"
