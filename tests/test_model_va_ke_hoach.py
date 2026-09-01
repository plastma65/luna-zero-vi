"""Khoá lại quy mô model và kế hoạch train đã chốt."""

from __future__ import annotations

import pytest

from luna_zero.checkpoint import TrainState
from luna_zero.config import MODEL, TRAIN

# sizing.py cố ý KHÔNG cần torch: phép tính quyết định model có vừa card hay không
# phải chạy được cả trên máy chưa cài torch.
from luna_zero.sizing import estimate_num_params, estimate_vram_gb
from luna_zero.train import con_lai, make_plan


def test_so_tham_so_quanh_110m() -> None:
    n = estimate_num_params()
    assert 100e6 <= n <= 125e6, f"model lệch khỏi cỡ 110M đã chốt: {n / 1e6:.1f}M"


def test_bo_buoc_embedding_lam_model_phinh() -> None:
    assert estimate_num_params(tie_embeddings=False) > estimate_num_params()


def test_ke_hoach_train_trong_khoang_ngay_chu_khong_phai_tuan() -> None:
    plan = make_plan()
    assert plan.tokens_per_step == (
        TRAIN.micro_batch_size * TRAIN.grad_accum_steps * MODEL.block_size
    )
    assert plan.total_steps * plan.tokens_per_step <= TRAIN.target_tokens
    # Khoảng này bám theo tốc độ ĐO THẬT 22k token/s trên 3060 (config đặt 20k làm biên),
    # cho ~1,3 ngày. Bản đầu canh 2-7 ngày vì dựa trên phỏng đoán 6.000 token/s.
    # Ra ngoài khoảng 1-3 ngày nghĩa là cấu hình hoặc phần cứng đã đổi -> phải xem lại.
    assert (
        1.0 <= plan.estimated_days <= 3.0
    ), f"ước lượng {plan.estimated_days:.2f} ngày, lệch khỏi kế hoạch đã đo"


# Trần VRAM cho test. Ngân sách thật KHÔNG phải 12GB: màn hình cắm vào chính 3060 nên
# Windows và trình duyệt đã ăn ~1,5GB, còn khoảng 10,5GB dùng được. Đo thật cho 6,91GB
# cấp phát / 8,16GB giữ chỗ. Đặt trần 9,0GB: đủ chỗ cho cấu hình hiện tại, và đỏ ngay
# nếu ai chỉnh batch hay block_size làm nó chạm mức nguy hiểm.
VRAM_TRAN_GB = 9.0


def test_vram_uoc_luong_nam_trong_tran() -> None:
    gb = estimate_vram_gb()
    assert gb <= VRAM_TRAN_GB, f"ước {gb:.2f}GB > trần {VRAM_TRAN_GB}GB — sẽ tràn card 3060 12GB"


def test_vram_phan_ung_dung_chieu_voi_block_size() -> None:
    """Phép đo chiều ngược: nhân đôi block_size PHẢI làm ước lượng tăng.

    Nếu không, hàm ước lượng vô dụng và test trần VRAM ở trên chỉ là trang trí.
    """
    from dataclasses import replace

    to_hon = replace(MODEL, block_size=MODEL.block_size * 2)
    assert estimate_vram_gb(to_hon) > estimate_vram_gb(MODEL) * 1.3


def test_chua_co_checkpoint_thi_con_nguyen_ke_hoach() -> None:
    plan = make_plan()
    steps, hours = con_lai(plan, None)
    assert steps == plan.total_steps
    assert hours == plan.estimated_hours


def test_co_checkpoint_thi_thoi_gian_con_lai_giam_theo() -> None:
    plan = make_plan()
    nua_duong = TrainState(step=plan.total_steps // 2, tokens_seen=0, data_position=0)
    steps, hours = con_lai(plan, nua_duong)
    assert steps == plan.total_steps - plan.total_steps // 2
    assert hours == pytest.approx(plan.estimated_hours / 2, rel=0.01)


def test_qua_ke_hoach_khong_ra_so_am() -> None:
    plan = make_plan()
    steps, hours = con_lai(
        plan, TrainState(step=plan.total_steps * 2, tokens_seen=0, data_position=0)
    )
    assert steps == 0 and hours == 0.0


# Số ĐO THẬT trên RTX 3060, cấu hình đã chốt, ngày 2026-09-02:
#   torch.cuda.max_memory_allocated() = 6.91 GB
VRAM_DO_THAT_GB = 6.91


def test_uoc_luong_vram_bam_sat_so_do_that() -> None:
    """Neo ước lượng vào phép đo thật, không chỉ vào một cái trần.

    `test_vram_uoc_luong_nam_trong_tran` chỉ bắt được ước lượng QUÁ CAO. Nhưng chiều
    nguy hiểm là QUÁ THẤP: hệ số kích hoạt 20 của bản đầu cho 4,46GB — nói "vừa card"
    trong khi thực tế 6,91GB. Nếu ai đó hạ hệ số xuống, ước lượng sẽ dễ chịu hơn và
    trần 9GB vẫn xanh, rồi lần train thật mới tràn VRAM.
    """
    uoc = estimate_vram_gb()
    lech = abs(uoc - VRAM_DO_THAT_GB) / VRAM_DO_THAT_GB
    assert lech <= 0.15, (
        f"ước {uoc:.2f}GB lệch {lech:.0%} so với số đo thật {VRAM_DO_THAT_GB}GB. "
        "Hoặc hệ số kích hoạt sai, hoặc cấu hình đã đổi và cần đo lại trên GPU."
    )
