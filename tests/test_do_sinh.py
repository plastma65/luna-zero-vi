"""Chỉ số lặp — biến "trông có vẻ ít lặp hơn" thành con số."""

from __future__ import annotations

from luna_zero.do_sinh import do_lap


def test_van_ban_khong_lap_cho_distinct_bang_mot() -> None:
    d = do_lap("một hai ba bốn năm sáu bảy tám")
    assert d.distinct_1 == 1.0
    assert d.distinct_2 == 1.0
    assert d.lap_dai_nhat == 1


def test_bat_dung_vong_lap_that() -> None:
    """Đúng chuỗi model đã sinh ra ở bước 2.069."""
    d = do_lap("bảo đảm, bảo đảm, bảo đảm, bảo đảm, bảo đảm")
    # 10 từ nhưng chỉ 3 từ khác nhau ("bảo", "đảm,", "đảm") -> distinct-1 = 0.30.
    # Ngưỡng đầu mình đặt < 0.30 và đỏ đúng ở biên. Dùng <= và ghi rõ con số thật,
    # thay vì nới ngưỡng cho qua — nới bừa là cách làm test mất giá trị.
    assert d.distinct_1 <= 0.30
    assert d.distinct_2 < 0.35
    assert d.lap_dai_nhat >= 2


def test_lap_nang_hon_thi_diem_thap_hon() -> None:
    """Phép đo phải xếp hạng ĐÚNG CHIỀU, không chỉ ra số nào đó."""
    it = do_lap("hôm nay trời nắng đẹp và gió nhẹ thổi qua khu vườn nhỏ")
    nhieu = do_lap("hôm nay hôm nay hôm nay hôm nay hôm nay hôm nay")
    assert nhieu.distinct_2 < it.distinct_2
    assert nhieu.lap_dai_nhat > it.lap_dai_nhat


def test_van_ban_rong_khong_no() -> None:
    d = do_lap("")
    assert d.n_token == 0 and d.distinct_1 == 0.0


def test_mot_tu_khong_no() -> None:
    d = do_lap("xin")
    assert d.n_token == 1
    assert d.distinct_1 == 1.0
    assert d.distinct_4 == 0.0


# --- neo vào văn bản người viết ---------------------------------------------
def test_khoang_tu_nhien_bao_quanh_so_do_duoc() -> None:
    from luna_zero.do_sinh import DISTINCT2_KHOANG_TU_NHIEN, DISTINCT2_NGUOI_VIET

    thap, cao = DISTINCT2_KHOANG_TU_NHIEN
    assert thap < DISTINCT2_NGUOI_VIET < cao


def test_lap_nang_bi_danh_dau_lech() -> None:
    from luna_zero.do_sinh import trong_khoang_tu_nhien

    assert not trong_khoang_tu_nhien(do_lap("bảo đảm, bảo đảm, bảo đảm, bảo đảm").distinct_2)


def test_da_dang_qua_muc_CUNG_bi_danh_dau_lech() -> None:
    """Phép đo hai chiều. distinct-2 = 1.000 KHÔNG phải điểm tuyệt đối.

    Văn bản người viết có distinct-2 quanh 0,853 — tức nó CÓ lặp. Model đạt 1,000 nghĩa
    là phạt lặp đã đè quá tay, không phải model viết hay hơn người. Nếu chỉ canh một
    chiều "càng cao càng tốt" thì ta lại tự lừa mình bằng một con số đẹp.
    """
    from luna_zero.do_sinh import trong_khoang_tu_nhien

    assert not trong_khoang_tu_nhien(1.0)
    assert not trong_khoang_tu_nhien(0.99)
