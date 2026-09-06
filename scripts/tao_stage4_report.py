"""Gom các artifact final đã khóa thành báo cáo Chặng 4 truy nguyên được."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.release import ghi_stage4_report, render_stage4_markdown  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", type=Path, default=config.ARTIFACT_DIR / "eval")
    p.add_argument("--output", type=Path, default=config.STAGE4_REPORT_PATH)
    args = p.parse_args()
    try:
        report = ghi_stage4_report(args.eval_dir, args.output)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    config.STAGE4_REPORT_MD_PATH.write_text(render_stage4_markdown(report), encoding="utf-8")
    loss = report["final_human_loss"]
    gen = report["final_generation"]
    print(f"Stage 4      : {report['status']}")
    print(f"Human final : NLL {loss['nll']:.4f} | PPL {loss['perplexity']:.3f}")
    print(
        f"Generation  : distinct-2 {gen['distinct_2_mean']:.3f} | "
        f"4-gram max {gen['max_4gram_repeat']}"
    )
    print(f"Đã lưu      : {args.output}")
    print(f"Bản đọc     : {config.STAGE4_REPORT_MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
