"""Tạo final held-out corpus từ một JSONL độc lập với corpus train/validation.

Script KHÔNG tự chọn hay tải dataset. Người chạy phải cung cấp một source bên ngoài
`data/raw`; script sẽ chứng minh exact document overlap = 0 trước khi ghi final.jsonl.

Ví dụ:
    python scripts/tao_eval_corpus.py --source D:/datasets/luna_eval_source.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.eval import tao_corpus_heldout  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="JSONL nguồn độc lập")
    p.add_argument("--field", default="text", help="tên trường chứa văn bản")
    p.add_argument("--output", type=Path, default=config.FINAL_EVAL_PATH)
    p.add_argument("--train-corpus-dir", type=Path, default=config.RAW_DIR)
    p.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="mặc định <output>.meta.json",
    )
    args = p.parse_args()

    meta_path = args.meta
    if meta_path is None:
        meta_path = args.output.with_suffix(".meta.json")

    try:
        ket_qua = tao_corpus_heldout(
            source=args.source,
            output=args.output,
            train_corpus_dir=args.train_corpus_dir,
            field=args.field,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "loai": "final_heldout_corpus",
        "gioi_han": (
            "Đã chứng minh không trùng nguyên document sau NFC với raw corpus cũ; "
            "chưa chứng minh không có near-duplicate/paraphrase."
        ),
        **asdict(ket_qua),
    }
    meta_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Đã tạo     : {args.output}")
    print(f"Document   : {ket_qua.n_docs:,}")
    print(f"Dung lượng : {ket_qua.n_bytes / 1024:.1f} KiB text")
    print(
        f"Đã quét    : {ket_qua.kiem_tra_tach_biet.n_train_docs_quet:,} doc corpus cũ; "
        "exact overlap = 0"
    )
    print(f"Fingerprint: {ket_qua.output_fingerprint}")
    print(f"Metadata   : {meta_path}")
    print("Giới hạn   : chưa loại được near-duplicate/paraphrase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
