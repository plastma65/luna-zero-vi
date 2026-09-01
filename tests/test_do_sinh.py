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
def test_khoang_hep_dan_khi_van_ban_dai_ra() -> None:
    """distinct-n phụ thuộc độ dài — đây là lý do phải tra bảng theo số từ.

    Bản đầu dùng MỘT mốc 0,853 lấy từ document đầy đủ rồi đem so với mẫu model 120 từ,
    và gắn nhãn LỆCH nhầm cho một mẫu bình thường.
    """
    from luna_zero.do_sinh import khoang_nguoi_viet

    _, tv_ngan, _ = khoang_nguoi_viet(60)
    _, tv_vua, _ = khoang_nguoi_viet(120)
    _, tv_dai, _ = khoang_nguoi_viet(5_000)
    assert tv_ngan > tv_vua > tv_dai, "văn bản dài hơn phải có distinct-2 thấp hơn"


def test_0963_o_do_dai_120_tu_la_TU_NHIEN() -> None:
    """Ca thật đã bị gắn nhãn sai: mẫu phở ở bước 11.304, 120 từ, distinct-2 0,963."""
    from luna_zero.do_sinh import trong_khoang_tu_nhien

    assert trong_khoang_tu_nhien(0.963, n_tu=120)
    # Cùng con số đó nhưng ở văn bản dài thì đúng là bất thường.
    assert not trong_khoang_tu_nhien(0.963, n_tu=5_000)


def test_lap_nang_bi_danh_dau_lech() -> None:
    from luna_zero.do_sinh import trong_khoang_tu_nhien

    assert not trong_khoang_tu_nhien(
        do_lap("bảo đảm, bảo đảm, bảo đảm, bảo đảm").distinct_2, n_tu=8
    )


def test_da_dang_qua_muc_CUNG_bi_danh_dau_lech() -> None:
    """Phép đo hai chiều. distinct-2 = 1.000 trên văn bản DÀI không phải điểm tuyệt đối:
    văn bản người viết có lặp, đè sạch là bất thường chứ không phải hay hơn."""
    from luna_zero.do_sinh import trong_khoang_tu_nhien

    assert not trong_khoang_tu_nhien(1.0, n_tu=5_000)
    assert not trong_khoang_tu_nhien(0.99, n_tu=400)
