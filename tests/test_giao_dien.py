"""Giao diện chat — ba chỗ hỏng ÂM THẦM, không chỗ nào ném lỗi.

1. Giải mã token lẻ làm vỡ chữ tiếng Việt thành "�" (trông như model viết bậy).
2. Bản `sinh_dan` dùng cho stream lệch với `sinh` dùng cho CLI (hai luật lấy mẫu).
3. Server trả cả cục ở cuối thay vì chảy dần — hiệu ứng "đang nghĩ" hết ý nghĩa và
   người dùng không phân biệt được model chậm với model treo.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from luna_zero import config
from luna_zero.giao_dien import GiaiMaDan, NguonGiaLap, NguonSinh, tao_server
from luna_zero.giao_dien.may_chu import ThamSoSinh

TRANG = Path(__file__).resolve().parents[1] / "src" / "luna_zero" / "giao_dien" / "trang.html"


@pytest.fixture
def server_gia_lap():
    """Server thật trên cổng do hệ điều hành cấp — cổng cố định sẽ đụng vòng train."""
    srv = tao_server(NguonGiaLap(nhip_giay=0.01, nghi_dau_giay=0.05), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _doc_dong(url: str, body: dict) -> Iterator[tuple[float, dict]]:
    """Gửi yêu cầu và yield (giây kể từ lúc gửi, gói JSON) cho từng dòng NDJSON."""
    req = urllib.request.Request(
        url + "/api/sinh",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    bat_dau = time.monotonic()
    with urllib.request.urlopen(req, timeout=30) as r:
        for dong in r:
            if dong.strip():
                yield time.monotonic() - bat_dau, json.loads(dong)


# --- 3. văn bản phải CHẢY, không phải rơi một cục -----------------------------
def test_chu_ve_dan_chu_khong_don_mot_cuc(server_gia_lap: str) -> None:
    """PHÉP ĐO CHIỀU NGƯỢC: đo thứ server KHÔNG được làm — gom hết rồi mới gửi.

    Nếu server đệm toàn bộ câu trả lời thì mọi dòng về gần như CÙNG một thời điểm.
    Test này bắt đúng điều đó bằng khoảng cách thời gian giữa dòng đầu và dòng cuối,
    nên không lách được bằng cách trả đúng nội dung.
    """
    moc = [t for t, goi in _doc_dong(server_gia_lap, {"moi": "Hà Nội"}) if goi["loai"] == "chu"]
    assert len(moc) > 5
    assert moc[-1] - moc[0] > 0.05, "mọi mẩu về cùng lúc -> server đang đệm cả cục"


def test_dong_cuoi_bao_xong_kem_chi_so(server_gia_lap: str) -> None:
    goi = [g for _, g in _doc_dong(server_gia_lap, {"moi": "Hà Nội"})]
    assert goi[-1]["loai"] == "xong"
    chi_so = goi[-1]["chi_so"]
    assert 0.0 <= chi_so["distinct_2"] <= 1.0
    assert chi_so["danh_gia"]


def test_thieu_cau_moi_thi_bao_400(server_gia_lap: str) -> None:
    req = urllib.request.Request(
        server_gia_lap + "/api/sinh",
        data=b'{"moi": "   "}',
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400


def test_trang_thai_cap_mac_dinh_cho_trinh_duyet(server_gia_lap: str) -> None:
    """HTML lấy giá trị thanh trượt từ đây. Nếu endpoint quên `mac_dinh` thì JS phải
    viết cứng số -> config.SINH có bản thứ hai bằng JavaScript (lỗi số 6)."""
    with urllib.request.urlopen(server_gia_lap + "/api/trang_thai", timeout=10) as r:
        t = json.load(r)
    assert t["mac_dinh"]["temperature"] == config.SINH.temperature
    assert t["mac_dinh"]["so_token"] == config.SINH.so_token


# Namespace của SVG là một URL nhưng trình duyệt KHÔNG tải nó — đây là ngoại lệ duy
# nhất được phép, và phải liệt kê tường minh để nó không thành cái cớ cho thẻ khác.
NGOAI_LE_URL = "http://www.w3.org/2000/svg"


def test_trang_html_tu_chua(server_gia_lap: str) -> None:
    """Máy đang train có thể không có mạng; một thẻ trỏ ra CDN là trang chết lúc đó.

    Đo bằng thứ trang KHÔNG được có: mọi thuộc tính kéo tài nguyên về (src, href, url())
    phải trỏ vào chính nó hoặc data:. Kiểm kiểu này không lách được bằng cách đổi tên CDN.
    """
    with urllib.request.urlopen(server_gia_lap + "/", timeout=10) as r:
        html = r.read().decode("utf-8")
    assert "Luna Zero" in html

    con_lai = html.replace(NGOAI_LE_URL, "")
    for cam in ("http://", "https://", "src="):
        assert cam not in con_lai, f"trang tham chiếu tài nguyên ngoài: {cam!r}"
    for href in re.findall(r"href=['\"]([^'\"]+)", con_lai):
        assert href.startswith("data:"), f"href kéo tài nguyên ngoài: {href[:40]!r}"


def test_mot_luot_sinh_mot_luc(server_gia_lap: str) -> None:
    """Hai lượt song song chia đôi tốc độ và nhân đôi VRAM trên GPU đang train."""
    ket_qua: list[int] = []

    def chay() -> None:
        try:
            for _ in _doc_dong(server_gia_lap, {"moi": "Hà Nội"}):
                pass
            ket_qua.append(200)
        except urllib.error.HTTPError as e:
            ket_qua.append(e.code)

    luong = [threading.Thread(target=chay) for _ in range(2)]
    for t in luong:
        t.start()
        time.sleep(0.05)
    for t in luong:
        t.join(timeout=30)
    assert sorted(ket_qua) == [200, 409]


# --- tham số từ trình duyệt không được tin ------------------------------------
def test_tham_so_hong_khong_lot_vao_vong_sinh() -> None:
    """`so_token` dạng chuỗi sẽ nổ giữa vòng sinh, tức là SAU khi header đã gửi đi —
    lúc đó không còn cách nào báo lỗi tử tế. Phải chặn ngay ở cửa."""
    with pytest.raises((ValueError, TypeError)):
        ThamSoSinh.tu_json({"so_token": "nhiều"})
    assert ThamSoSinh.tu_json({"so_token": 999_999}).so_token <= 1000
    assert ThamSoSinh.tu_json({"lung_tung": 1}) == ThamSoSinh()


# --- 1. giải mã dần không được vỡ chữ tiếng Việt ------------------------------
def test_giai_ma_dan_khong_de_ra_ky_tu_rac(tiny_tokenizer) -> None:
    """Ghép mọi mẩu phát ra phải bằng ĐÚNG bản decode một lần, và không có "�".

    Đây là phép đo chiều ngược: nó bắt cả trường hợp mẩu vẫn "đọc được" nhưng thiếu
    dấu — thứ mà mắt lướt qua rất dễ bỏ sót trên màn hình chat.
    """
    goc = "Ngày mai trời sẽ đẹp, mùa xuân đã về trên những cánh đồng."
    ids = tiny_tokenizer.encode(goc)
    assert len(ids) > 10, "fixture phải cắt ra nhiều token thì test mới có ý nghĩa"

    giai_ma = GiaiMaDan(tiny_tokenizer)
    mau = [giai_ma.them(i) for i in ids]
    ghep = "".join(mau)

    assert "�" not in ghep
    assert ghep == tiny_tokenizer.decode(ids)
    assert sum(1 for m in mau if m == "") < len(mau), "không phát gì cho tới cuối"


def test_giai_ma_dan_giu_lai_duoi_do_dang(tiny_tokenizer) -> None:
    """Cố tình nạp nửa chuỗi UTF-8: bước đó phải im lặng chờ, không phát ký tự rác."""
    ids = tiny_tokenizer.encode("đường")
    giai_ma = GiaiMaDan(tiny_tokenizer)
    for i in ids:
        assert "�" not in giai_ma.them(i)
    assert giai_ma.van_ban == tiny_tokenizer.decode(ids)


# --- 2. stream và CLI phải là MỘT luật lấy mẫu --------------------------------
def test_sinh_dan_va_sinh_cho_cung_ket_qua() -> None:
    """Cùng seed thì hai lối gọi phải ra ĐÚNG một chuỗi. Lệch nghĩa là luật lấy mẫu đã
    có hai bản — đúng lỗi số 6, chỉ khác là lần này nó nằm trong xác suất nên chỉ lộ
    ra dưới dạng "giao diện viết khác terminal"."""
    torch = pytest.importorskip("torch", reason="cần torch")
    from dataclasses import replace

    from luna_zero.model import LunaZeroGPT

    cfg = replace(config.MODEL, n_layer=2, d_model=64, n_head=4, block_size=32, vocab_size=97)
    torch.manual_seed(0)
    m = LunaZeroGPT(cfg)
    x = torch.tensor([[config.BOS_ID, 5, 7]], dtype=torch.long)

    torch.manual_seed(123)
    mot_cuc = m.sinh(x, max_new_tokens=20, top_p=0.9, phat_lap=1.05, dung_o_eos=False)

    torch.manual_seed(123)
    tung_cai = x
    for tiep in m.sinh_dan(x, max_new_tokens=20, top_p=0.9, phat_lap=1.05, dung_o_eos=False):
        tung_cai = torch.cat([tung_cai, tiep], dim=1)

    assert torch.equal(mot_cuc, tung_cai)


def test_nguon_gia_lap_dung_giao_uoc() -> None:
    """Bản giả lập phải là NguonSinh thật, không phải đồ chơi riêng — nếu nó lệch giao
    ước thì giao diện chạy ngon lúc giả lập rồi vỡ đúng lúc gắn checkpoint thật."""
    nguon = NguonGiaLap(nhip_giay=0.0, nghi_dau_giay=0.0)
    assert isinstance(nguon, NguonSinh)
    assert set(nguon.trang_thai()) == {"che_do", "buoc", "val_loss", "thiet_bi"}
    assert "".join(nguon.sinh("Hà Nội", ThamSoSinh())).strip()


# --- trang nói đúng model làm được gì ---------------------------------------
def test_trang_noi_ro_day_la_model_nen() -> None:
    """Bố cục bong bóng chat mời gọi hỏi đáp mạnh hơn một dòng chữ giải thích.

    Người dùng đầu tiên gõ ngay "hà nội là ?" và nhận về một đoạn viết tiếp — model
    làm đúng việc của nó, nhưng giao diện đã hứa sai. Trang phải nói thẳng.
    """
    trang = (TRANG).read_text(encoding="utf-8")
    assert "model nền" in trang
    assert "chưa được dạy trả lời câu" in trang


def test_co_cau_mo_dau_bam_duoc() -> None:
    """Dạy bằng tay, không bằng chữ: bấm một cái ra ngay câu ĐÚNG DẠNG."""
    trang = (TRANG).read_text(encoding="utf-8")
    assert 'id="goi-y"' in trang
    assert "Hà Nội là thủ đô của" in trang
    # Chip phải điền vào ô nhập chứ KHÔNG gửi luôn — người dùng còn sửa được.
    assert "oNhap.value = b.textContent" in trang
    assert "goi-y" in trang.split("addEventListener")[1] or 'getElementById("goi-y")' in trang


def test_nhac_khi_go_cau_hoi() -> None:
    """Nhắc đúng lúc hiểu nhầm xảy ra, không phải cảnh báo chung ai cũng lướt qua."""
    trang = (TRANG).read_text(encoding="utf-8")
    assert 'id="nhac"' in trang
    assert 'endsWith("?")' in trang
    assert "TU_HOI" in trang


def test_van_tu_chua_khong_goi_ra_ngoai() -> None:
    """Phép đo chiều ngược, giữ nguyên giao ước ban đầu của trang.

    Máy đang train có thể không có mạng. Một trang phụ thuộc CDN sẽ hỏng đúng lúc đó.
    """
    trang = (TRANG).read_text(encoding="utf-8")
    assert 'src="http' not in trang
    assert 'href="http' not in trang
    assert "@import" not in trang
