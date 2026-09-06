"""Xuất checkpoint Luna Zero final thành package inference-only cho Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.checkpoint import CheckpointManager  # noqa: E402
from luna_zero.release import load_hf_package_cpu, xuat_hf_package  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    p.add_argument("--eval-dir", type=Path, default=config.ARTIFACT_DIR / "eval")
    p.add_argument("--output-dir", type=Path, default=config.RELEASE_DIR)
    args = p.parse_args()

    ckpt = CheckpointManager(args.checkpoint_dir).latest_path()
    if ckpt is None:
        print(f"Không tìm thấy checkpoint trong {args.checkpoint_dir}", file=sys.stderr)
        return 1
    pattern = f"generation_final_step_{config.SINH_EVAL.final_checkpoint_step:07d}_*.json"
    gen = sorted(args.eval_dir.glob(pattern))
    if len(gen) != 1:
        print(f"Cần đúng 1 generation final artifact, tìm thấy {len(gen)}.", file=sys.stderr)
        return 2
    try:
        package = xuat_hf_package(ckpt, config.STAGE4_REPORT_PATH, gen[0], args.output_dir)
        load_hf_package_cpu(package)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Checkpoint : {ckpt}")
    print(f"Package    : {package}")
    print("CPU smoke  : PASS — load từ model.safetensors và forward thành công")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
