"""Khóa corpus human-authored cho phép đo NLL/PPL cuối của Luna Zero.

Nguồn phải được chuẩn bị bên ngoài data/raw và chưa từng dùng để train, chọn checkpoint
hay tune sampling. Script yêu cầu provenance tường minh, quét exact document overlap
trên toàn raw corpus, rồi mới ghi final_human.jsonl + metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.eval import ProvenanceHuman, tao_corpus_human_heldout  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-name", required=True)
    p.add_argument("--source-url", required=True)
    p.add_argument("--retrieved-at", required=True, help="ngày lấy nguồn, ví dụ 2026-09-06")
    p.add_argument("--license-or-permission", required=True)
    p.add_argument("--notes", default="")
    p.add_argument("--field", default="text")
    p.add_argument("--train-corpus-dir", type=Path, default=config.RAW_DIR)
    p.add_argument("--xac-nhan-human-authored", action="store_true")
    args = p.parse_args()

    provenance = ProvenanceHuman(
        source_name=args.source_name,
        source_url=args.source_url,
        retrieved_at=args.retrieved_at,
        license_or_permission=args.license_or_permission,
        notes=args.notes,
    )
    try:
        report = tao_corpus_human_heldout(
            source=args.source,
            output=config.FINAL_HUMAN_EVAL_PATH,
            train_corpus_dir=args.train_corpus_dir,
            provenance=provenance,
            human_authored_confirmed=args.xac_nhan_human_authored,
            field=args.field,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    config.FINAL_HUMAN_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.FINAL_HUMAN_META_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    kiem = report["kiem_tra_tach_biet"]
    print(f"Đã khóa     : {config.FINAL_HUMAN_EVAL_PATH}")
    print(f"Document    : {report['n_docs']:,}")
    print(f"Đã quét     : {kiem['n_train_docs_quet']:,} doc raw; exact overlap = 0")
    print(f"Fingerprint : {report['output_fingerprint']}")
    print(f"Provenance  : {config.FINAL_HUMAN_META_PATH}")
    print("Giới hạn    : chưa loại được near-duplicate/paraphrase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
