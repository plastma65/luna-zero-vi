"""Khoá lại quy mô model và kế hoạch train đã chốt."""

from __future__ import annotations

import pytest

# model.py import torch ở mức module. Bỏ qua cả file khi máy chưa cài torch, thay vì
# làm hỏng việc thu thập test và kéo sập toàn bộ suite.
pytest.importorskip("torch", reason="model.py cần torch")

from luna_zero.checkpoint import TrainState  # noqa: E402
from luna_zero.config import MODEL, TRAIN  # noqa: E402
from luna_zero.model import estimate_num_params, estimate_vram_gb  # noqa: E402
from luna_zero.train import con_lai, make_plan  # noqa: E402


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
    assert 2 <= plan.estimated_days <= 7, f"ước lượng {plan.estimated_days:.1f} ngày, lệch kế hoạch"


# Trần VRAM cho test: card 12GB, mục tiêu ~4,5GB. Đặt 6,0GB để còn biên cho phân mảnh
# bộ nhớ và eval xen kẽ. Vượt trần nghĩa là cấu hình đã rời khỏi kế hoạch đã chốt.
VRAM_TRAN_GB = 6.0


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
