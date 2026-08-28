"""Khung vòng lặp huấn luyện.

Chặng 1 mới có phần lập kế hoạch: từ config suy ra số bước và thời gian chạy dự kiến.
Việc này để test khoá lại — nếu ai đó chỉnh batch/block_size làm kế hoạch nhảy sang
hàng tuần thay vì hàng ngày thì phải thấy ngay, chứ không phải sau 3 ngày train.
"""

from __future__ import annotations

from dataclasses import dataclass

from luna_zero.checkpoint import TrainState, make_manager
from luna_zero.config import MODEL, TRAIN, ModelConfig, TrainConfig


@dataclass(frozen=True)
class TrainingPlan:
    tokens_per_step: int
    total_steps: int
    estimated_hours: float

    @property
    def estimated_days(self) -> float:
        return self.estimated_hours / 24


def make_plan(model: ModelConfig = MODEL, train: TrainConfig = TRAIN) -> TrainingPlan:
    tokens_per_step = train.micro_batch_size * train.grad_accum_steps * model.block_size
    total_steps = train.target_tokens // tokens_per_step
    hours = train.target_tokens / train.tokens_per_second_3060 / 3600
    return TrainingPlan(
        tokens_per_step=tokens_per_step,
        total_steps=total_steps,
        estimated_hours=hours,
    )


def con_lai(plan: TrainingPlan, state: TrainState | None) -> tuple[int, float]:
    """Còn bao nhiêu bước và bao nhiêu giờ nữa, tính từ checkpoint đang có."""
    da_lam = state.step if state else 0
    steps = max(0, plan.total_steps - da_lam)
    return steps, plan.estimated_hours * steps / plan.total_steps


def main() -> None:
    plan = make_plan()
    print(f"Token mỗi bước : {plan.tokens_per_step:,}")
    print(f"Tổng số bước   : {plan.total_steps:,}")
    print(f"Thời gian ước  : {plan.estimated_days:.1f} ngày")
    print(
        f"Lưu checkpoint : mỗi {TRAIN.save_every_steps} bước, giữ {TRAIN.max_checkpoints_keep} bản"
    )

    da_co = make_manager().load_latest()
    state = da_co[0] if da_co else None
    steps, hours = con_lai(plan, state)
    if state is None:
        print("Trạng thái     : chưa có checkpoint, sẽ train từ đầu")
    else:
        print(
            f"Trạng thái     : chạy tiếp từ bước {state.step:,} " f"(token {state.data_position:,})"
        )
    print(f"Còn lại        : {steps:,} bước / {hours / 24:.2f} ngày")
    print("Vòng lặp train sẽ được viết ở Chặng 3.")


if __name__ == "__main__":
    main()
