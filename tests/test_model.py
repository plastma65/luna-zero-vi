"""Model GPT và bộ nạp batch — ba lỗi cho loss ĐẸP GIẢ.

Cả ba đều không ném lỗi, không cảnh báo, và làm loss trông tốt hơn sự thật:
1. Nhãn lệch một bước -> model học chép token hiện tại, loss lao về 0.
2. Attention nhìn được tương lai -> model đọc trộm đáp án.
3. Khởi tạo sai -> loss bắt đầu ở chỗ vô nghĩa và train không bao giờ hồi.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="cần torch")

from luna_zero.config import MODEL, TRAIN  # noqa: E402
from luna_zero.loader import BatchLoader  # noqa: E402
from luna_zero.model import LunaZeroGPT, estimate_num_params  # noqa: E402
from luna_zero.train import lr_tai_buoc, make_plan  # noqa: E402

TINY = replace(MODEL, n_layer=2, d_model=64, n_head=4, block_size=32, vocab_size=97)


@pytest.fixture
def tiny_model() -> LunaZeroGPT:
    torch.manual_seed(0)
    return LunaZeroGPT(TINY)


# --- kích cỡ ---------------------------------------------------------------
def test_so_tham_so_khop_cong_thuc() -> None:
    """Công thức đếm và mạng thật phải ra cùng một số, nếu không thì mọi ước lượng
    VRAM và thời gian đều dựa trên con số sai."""
    m = LunaZeroGPT(TINY)
    assert m.so_tham_so() == estimate_num_params(TINY)


def test_model_that_dung_110m() -> None:
    m = LunaZeroGPT(MODEL)
    n = m.so_tham_so()
    assert 100e6 <= n <= 125e6, f"{n / 1e6:.1f}M lệch khỏi cỡ đã chốt"


def test_embedding_duoc_buoc_chung(tiny_model: LunaZeroGPT) -> None:
    """Buộc chung wte và lm_head tiết kiệm 22% tham số. Tháo ra là model phình lên."""
    assert tiny_model.transformer["wte"].weight is tiny_model.lm_head.weight


# --- khởi tạo ---------------------------------------------------------------
def test_loss_ban_dau_bang_ln_vocab(tiny_model: LunaZeroGPT) -> None:
    """Model chưa học gì phải đoán đều tay -> loss = ln(vocab_size).

    Lệch xa nghĩa là khởi tạo sai. Đây là phép kiểm rẻ nhất và bắt được nhiều lỗi nhất
    trước khi đốt bốn ngày GPU.

    NHÃN PHẢI ĐỘC LẬP VỚI ĐẦU VÀO. Bản đầu của test này truyền `targets=x`, tức bắt model
    dự đoán chính token đang nhìn, và ra loss 3,87 thay vì 4,57. Không phải model sai:
    embedding buộc chung khiến `logits = h @ E^T`, mà tích vô hướng của một embedding với
    CHÍNH NÓ luôn lớn hơn với embedding khác — nên ngay lúc khởi tạo model đã đoán token
    hiện tại giỏi hơn ngẫu nhiên. Đúng bằng cơ chế của lỗi lệch-một-bước mà
    `test_nhan_la_dau_vao_dich_dung_mot_buoc` sinh ra để bắt.
    """
    torch.manual_seed(0)
    x = torch.randint(0, TINY.vocab_size, (4, TINY.block_size))
    y = torch.randint(0, TINY.vocab_size, (4, TINY.block_size))
    loss = tiny_model(x, y).loss.item()
    ky_vong = math.log(TINY.vocab_size)
    assert abs(loss - ky_vong) < 0.35, f"loss ban đầu {loss:.3f}, kỳ vọng ~{ky_vong:.3f}"


def test_nhan_trung_dau_vao_cho_loss_thap_bat_thuong(tiny_model: LunaZeroGPT) -> None:
    """Khoá lại chính hiện tượng vừa phát hiện, để nó không bị hiểu nhầm lần nữa.

    Nếu ai đó thấy loss thấp hơn ln(vocab) rõ rệt lúc mới train, đây là một nguyên nhân
    có thật cần loại trừ trước khi mừng: nhãn đang trùng đầu vào.
    """
    torch.manual_seed(0)
    x = torch.randint(0, TINY.vocab_size, (4, TINY.block_size))
    y = torch.randint(0, TINY.vocab_size, (4, TINY.block_size))
    loss_trung = tiny_model(x, x).loss.item()
    loss_doc_lap = tiny_model(x, y).loss.item()
    assert loss_trung < loss_doc_lap - 0.3, (
        "nhãn trùng đầu vào phải cho loss thấp hơn rõ rệt; nếu không, "
        "embedding có thể đã bị tháo buộc chung"
    )


# --- nhân quả ---------------------------------------------------------------
def test_khong_nhin_duoc_tuong_lai(tiny_model: LunaZeroGPT) -> None:
    """Phép đo chiều ngược quan trọng nhất của model.

    Đổi token ở vị trí t KHÔNG được làm thay đổi logits ở mọi vị trí < t. Rò rỉ tương
    lai cho loss rất đẹp mà model sinh văn bản thì vô dụng — không có cách nào phát
    hiện bằng cách nhìn đường loss.
    """
    tiny_model.eval()
    torch.manual_seed(0)
    x = torch.randint(0, TINY.vocab_size, (1, TINY.block_size))
    t = TINY.block_size // 2
    with torch.no_grad():
        a = tiny_model(x).logits
        x2 = x.clone()
        x2[0, t] = (x2[0, t] + 1) % TINY.vocab_size
        b = tiny_model(x2).logits
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-5), "attention rò rỉ tương lai"
    assert not torch.allclose(a[:, t], b[:, t]), "đổi token phải ảnh hưởng chính vị trí đó"


def test_chuoi_dai_hon_block_size_bi_chan(tiny_model: LunaZeroGPT) -> None:
    x = torch.zeros((1, TINY.block_size + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="block_size"):
        tiny_model(x)


# --- chạy được trên CPU -----------------------------------------------------
def test_chay_duoc_tren_cpu() -> None:
    """Điểm bán hàng của Luna Zero so với Luna cũ. CI GitHub không có GPU nên test này
    là lưới chặn thật, không phải lời hứa suông."""
    m = LunaZeroGPT(TINY).to("cpu")
    x = torch.randint(0, TINY.vocab_size, (2, 8))
    out = m(x, x)
    assert out.logits.device.type == "cpu"
    assert out.loss.item() > 0


def test_sinh_van_ban_dung_hinh_dang(tiny_model: LunaZeroGPT) -> None:
    x = torch.zeros((1, 3), dtype=torch.long)
    y = tiny_model.sinh(x, max_new_tokens=5, top_k=10)
    assert y.shape == (1, 8)
    assert y[:, :3].tolist() == x.tolist(), "phần ngữ cảnh gốc phải giữ nguyên"


def test_sinh_khong_no_khi_vuot_block_size(tiny_model: LunaZeroGPT) -> None:
    """Ngữ cảnh dài hơn block_size phải được cắt, không được ném lỗi."""
    x = torch.zeros((1, TINY.block_size), dtype=torch.long)
    y = tiny_model.sinh(x, max_new_tokens=3)
    assert y.shape[1] == TINY.block_size + 3


# --- optimizer --------------------------------------------------------------
def test_layernorm_va_bias_khong_bi_weight_decay(tiny_model: LunaZeroGPT) -> None:
    """Phạt weight decay lên LayerNorm là kéo hệ số chuẩn hoá về 0 — hỏng đúng thứ
    giữ cho train ổn định. Lỗi kinh điển và hoàn toàn im lặng."""
    nhom = tiny_model.nhom_tham_so_optimizer(0.1)
    assert nhom[0]["weight_decay"] == 0.1
    assert nhom[1]["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in nhom[0]["params"])
    assert all(p.dim() < 2 for p in nhom[1]["params"])
    tong = sum(p.numel() for g in nhom for p in g["params"])
    assert tong == tiny_model.so_tham_so()


# --- lịch learning rate -----------------------------------------------------
def test_warmup_tang_dan_tu_gan_khong() -> None:
    plan = make_plan()
    lr0 = lr_tai_buoc(0, plan)
    lr_giua = lr_tai_buoc(TRAIN.warmup_steps // 2, plan)
    lr_dinh = lr_tai_buoc(TRAIN.warmup_steps, plan)
    assert lr0 < lr_giua < lr_dinh
    assert lr_dinh == pytest.approx(TRAIN.learning_rate, rel=0.02)


def test_lr_khong_bao_gio_vuot_dinh_va_ket_thuc_o_min() -> None:
    plan = make_plan()
    moc = [0, 1, TRAIN.warmup_steps, plan.total_steps // 2, plan.total_steps - 1]
    assert all(lr_tai_buoc(s, plan) <= TRAIN.learning_rate + 1e-12 for s in moc)
    assert lr_tai_buoc(plan.total_steps, plan) == pytest.approx(TRAIN.min_lr)
    assert lr_tai_buoc(plan.total_steps * 2, plan) == pytest.approx(TRAIN.min_lr)


# --- bộ nạp batch -----------------------------------------------------------
@pytest.fixture
def bin_gia(tmp_path: Path) -> Path:
    """File .bin tí hon với token tăng dần: kiểm quan hệ dịch bằng mắt được."""
    arr = np.arange(1000, dtype=np.uint16)
    (tmp_path / "train.bin").write_bytes(arr.tobytes())
    return tmp_path


def test_nhan_la_dau_vao_dich_dung_mot_buoc(bin_gia: Path) -> None:
    """Lỗi lệch một bước làm model học chép token hiện tại, loss lao xuống gần 0
    và trông như thành công vang dội. Đây là lưới chặn."""
    ld = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=0)
    x, y = ld.lay_batch(4)
    assert torch.equal(x[:, 1:], y[:, :-1]), "y phải là x dịch phải đúng một vị trí"
    # Dữ liệu là dãy tăng dần nên y = x + 1 ở mọi vị trí.
    assert torch.equal(y, x + 1)


def test_cua_so_khong_vuot_bien_file(bin_gia: Path) -> None:
    """Lấy mẫu ở cuối file mà không chừa block_size+1 sẽ đọc lẫn vùng rác."""
    ld = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=1)
    for _ in range(50):
        x, y = ld.lay_batch(16)
        assert int(y.max()) <= 999, "đọc vượt quá token cuối cùng của file"
        assert int(x.min()) >= 0


def test_file_qua_nho_bao_loi_ro_rang(tmp_path: Path) -> None:
    (tmp_path / "train.bin").write_bytes(np.arange(4, dtype=np.uint16).tobytes())
    with pytest.raises(ValueError, match="cần hơn"):
        BatchLoader(tmp_path, "train", block_size=32, device="cpu")


def test_thieu_file_bao_loi_chi_ro_viec_can_lam(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="dong_goi"):
        BatchLoader(tmp_path, "train", block_size=8, device="cpu")


def test_cung_seed_cho_cung_batch(bin_gia: Path) -> None:
    a = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=7).lay_batch(4)
    b = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=7).lay_batch(4)
    assert torch.equal(a[0], b[0])


def test_seed_khac_cho_thu_tu_khac(bin_gia: Path) -> None:
    a = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=1).lay_batch(8)
    b = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=2).lay_batch(8)
    assert not torch.equal(a[0], b[0])


# --- duyệt hoán vị: độ phủ và chạy tiếp -------------------------------------
def test_moi_cua_so_duoc_thay_dung_mot_lan_trong_mot_vong(bin_gia: Path) -> None:
    """Phép đo chiều ngược cho độ phủ dữ liệu.

    Lấy mẫu ngẫu nhiên CÓ HOÀN LẠI chỉ chạm tới ~62% corpus khi số token lấy ra xấp xỉ
    kích thước corpus — gần 38% dữ liệu không bao giờ được model nhìn thấy, mà không có
    gì báo. Duyệt hoán vị thì mỗi cửa sổ đúng một lần.
    """
    ld = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=3)
    n = len(ld)
    thay = []
    for _ in range(n):
        x, _ = ld.lay_batch(1)
        thay.append(int(x[0, 0]))  # dữ liệu tăng dần -> token đầu = vị trí bắt đầu
    assert len(set(thay)) == n, "có cửa sổ bị lặp trong cùng một vòng"
    assert sorted(thay) == [i * 8 for i in range(n)], "có cửa sổ bị bỏ sót"


def test_het_mot_vong_thi_tron_lai_chu_khong_dung(bin_gia: Path) -> None:
    ld = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=3)
    n = len(ld)
    vong1 = [int(ld.lay_batch(1)[0][0, 0]) for _ in range(n)]
    vong2 = [int(ld.lay_batch(1)[0][0, 0]) for _ in range(n)]
    assert sorted(vong1) == sorted(vong2), "vòng hai phải phủ cùng tập cửa sổ"
    assert vong1 != vong2, "vòng hai phải có thứ tự khác"


def test_chay_tiep_dung_cho_da_dung(bin_gia: Path) -> None:
    """Cốt lõi của việc ngắt giữa chừng: nạp lại từ data_position phải cho ĐÚNG batch
    tiếp theo, không lặp lại và không nhảy cóc."""
    a = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=5)
    a.lay_batch(4)
    a.lay_batch(4)
    vi_tri = a.vi_tri
    mong_doi, _ = a.lay_batch(4)

    b = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=5, vi_tri=vi_tri)
    thuc_te, _ = b.lay_batch(4)
    assert torch.equal(mong_doi, thuc_te)


def test_vi_tri_tang_dung_bang_batch_size(bin_gia: Path) -> None:
    ld = BatchLoader(bin_gia, "train", block_size=8, device="cpu")
    ld.lay_batch(6)
    assert ld.vi_tri == 6
    ld.lay_batch(3)
    assert ld.vi_tri == 9


def test_chay_tiep_qua_ranh_gioi_vong(bin_gia: Path) -> None:
    """Điểm dừng rơi đúng lúc hết một vòng là ca dễ sai nhất."""
    ld = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=9)
    n = len(ld)
    a = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=9, vi_tri=n - 2)
    mong_doi, _ = a.lay_batch(4)
    b = BatchLoader(bin_gia, "train", block_size=8, device="cpu", seed=9, vi_tri=n - 2)
    thuc_te, _ = b.lay_batch(4)
    assert torch.equal(mong_doi, thuc_te)
    assert a.vi_tri == n + 2


def test_kieu_du_lieu_la_int64_cho_embedding(bin_gia: Path) -> None:
    """Token lưu uint16 nhưng nn.Embedding đòi int64. Quên ép kiểu là nổ lúc chạy."""
    x, y = BatchLoader(bin_gia, "train", block_size=8, device="cpu").lay_batch(2)
    assert x.dtype == torch.int64 and y.dtype == torch.int64


# --- sinh văn bản từ checkpoint ---------------------------------------------
def test_dung_lai_model_theo_cau_hinh_trong_checkpoint(tmp_path: Path) -> None:
    """Phép đo chiều ngược: nạp trọng số PHẢI theo cấu hình đã lưu, không theo config
    hiện tại.

    Nếu ai sửa config.py sau khi train (đổi n_layer, d_model...), nạp theo config mới
    sẽ lệch hình dạng và nổ — hoặc tệ hơn, khớp hình dạng nhưng sai ý nghĩa và model
    sinh ra chữ rác mà không báo gì.
    """
    from dataclasses import asdict

    goc = LunaZeroGPT(TINY)
    blob = {"model": goc.state_dict(), "model_cfg": asdict(TINY)}

    khac = replace(MODEL, n_layer=6, d_model=128)  # "config hiện tại" đã bị sửa
    dung = replace(khac, **blob["model_cfg"])
    assert dung == TINY

    moi = LunaZeroGPT(dung)
    moi.load_state_dict(blob["model"])  # không được ném
    x = torch.randint(0, TINY.vocab_size, (1, 4))
    goc.eval()
    moi.eval()
    with torch.no_grad():
        assert torch.allclose(goc(x).logits, moi(x).logits, atol=1e-6)


def test_nap_theo_config_hien_tai_se_no(tmp_path: Path) -> None:
    """Chứng minh chiều ngược lại thật sự hỏng — nếu không thì test trên là trang trí."""
    goc = LunaZeroGPT(TINY)
    khac = replace(TINY, n_layer=6)
    with pytest.raises(RuntimeError):
        LunaZeroGPT(khac).load_state_dict(goc.state_dict())


# --- lấy mẫu khi sinh -------------------------------------------------------
def test_top_p_giu_it_nhat_mot_token(tiny_model: LunaZeroGPT) -> None:
    """top_p rất nhỏ vẫn phải sinh được: nếu lọc sạch mọi token thì multinomial nổ."""
    x = torch.zeros((1, 3), dtype=torch.long)
    y = tiny_model.sinh(x, max_new_tokens=5, top_p=0.01)
    assert y.shape == (1, 8)


def test_phat_lap_ha_diem_token_da_xuat_hien(tiny_model: LunaZeroGPT) -> None:
    """Phép đo chiều ngược cho phạt lặp: token đã có PHẢI bị hạ điểm, không phải nâng.

    Chỗ này rất dễ sai: chia đều mọi logit cho hệ số phạt sẽ làm điểm ÂM to lên
    (âm chia cho 1.15 thì gần 0 hơn) và biến hình phạt thành phần thưởng.
    """
    tiny_model.eval()
    torch.manual_seed(0)
    x = torch.randint(0, TINY.vocab_size, (1, 16))
    with torch.no_grad():
        goc = tiny_model(x).logits[:, -1, :].clone()

    da_co = torch.unique(x[0])
    diem = goc[0, da_co]
    phat = 1.5
    sau = torch.where(diem < 0, diem * phat, diem / phat)
    assert torch.all(sau <= diem + 1e-6), "phạt lặp đang nâng điểm thay vì hạ"


def test_sinh_khong_lap_vo_han_voi_phat_lap(tiny_model: LunaZeroGPT) -> None:
    """Chỉ kiểm hành vi: có phạt lặp thì số token khác nhau không được ít hơn hẳn."""
    torch.manual_seed(0)
    x = torch.zeros((1, 2), dtype=torch.long)
    khong_phat = tiny_model.sinh(x, max_new_tokens=60, top_p=0.9, phat_lap=1.0)
    torch.manual_seed(0)
    co_phat = tiny_model.sinh(x, max_new_tokens=60, top_p=0.9, phat_lap=1.3)
    assert len(set(co_phat[0].tolist())) >= len(set(khong_phat[0].tolist()))
