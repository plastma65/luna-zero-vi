"""Smoke test package Hugging Face Luna Zero hoàn toàn từ artifact inference-only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.release import load_hf_package_cpu  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "package",
        nargs="?",
        type=Path,
        default=config.RELEASE_DIR / config.RELEASE.package_dir_name,
    )
    args = p.parse_args()
    try:
        load_hf_package_cpu(args.package)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"PASS: load CPU + forward từ package {args.package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
