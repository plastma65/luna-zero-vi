"""Train Luna Zero.

python scripts/train_model.py                 # chạy thật, 3-5 ngày, Ctrl+C được
python scripts/train_model.py --smoke         # 20 bước trên model tí hon, vài giây
python scripts/train_model.py --device cpu    # ép chạy CPU (chỉ để kiểm, cực chậm)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.train import con_lai, make_plan, train_loop  # noqa: E402

# Model tí hon cho smoke test: đủ nhỏ để chạy trên CPU trong vài giây, nhưng đi qua
# ĐÚNG những đường code mà bản thật đi qua.
SMOKE_LAYER, SMOKE_DMODEL, SMOKE_HEAD, SMOKE_BLOCK = 2, 64, 2, 64
SMOKE_STEPS = 20


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--tokenizer", type=Path, default=config.TOKENIZER_PATH)
    p.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    p.add_argument("--device", default=None, help="cuda | cpu. Mặc định tự dò.")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    model_cfg = config.MODEL
    train_cfg = config.TRAIN
    max_steps = args.max_steps
    if args.smoke:
        model_cfg = replace(
            model_cfg,
            n_layer=SMOKE_LAYER,
            d_model=SMOKE_DMODEL,
            n_head=SMOKE_HEAD,
            block_size=SMOKE_BLOCK,
        )
        train_cfg = replace(train_cfg, micro_batch_size=2, grad_accum_steps=2, warmup_steps=5)
        max_steps = max_steps or SMOKE_STEPS
        args.checkpoint_dir = args.checkpoint_dir / "smoke"

    plan = make_plan(model_cfg, train_cfg)
    steps, hours = con_lai(plan, None)
    # Khi có --max-steps, thời gian phải co lại theo. Bản đầu in "20 bước, ước 4,2 ngày"
    # vì lấy giờ của kế hoạch đầy đủ — con số đúng đặt cạnh con số sai thì cả hai mất tin.
    if max_steps is not None:
        steps = min(steps, max_steps)
        hours = plan.estimated_hours * steps / plan.total_steps
    don_vi = f"{hours * 60:.1f} phút" if hours < 1 else f"{hours / 24:.2f} ngày"
    print(f"Kế hoạch : {steps:,} bước, ước {don_vi}")

    state = train_loop(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer,
        checkpoint_dir=args.checkpoint_dir,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        device=args.device,
        max_steps=max_steps,
    )
    print(f"\nDừng ở bước {state.step:,} | val loss tốt nhất {state.best_val_loss:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
