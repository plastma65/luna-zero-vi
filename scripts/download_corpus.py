"""Tải corpus tiếng Việt về data/raw/*.jsonl.

Dùng:
    python scripts/download_corpus.py --source wikipedia --target-gb 1.5
    python scripts/download_corpus.py --source culturax  --target-gb 6

Ghi chú về nguồn:
* wikipedia  — `wikimedia/wikipedia`, config `20231101.vi`. KHÔNG cần đăng nhập.
               Sạch nhất, ~1,5GB. Đủ dư để train tokenizer ở Chặng 1.
* culturax   — `uonlp/CulturaX`, subset `vi`. Đã lọc sẵn, đây là nguồn chính cho
               2,2 tỷ token. CÓ GATE: phải đăng nhập HF và bấm đồng ý điều khoản
               trên trang dataset trước, rồi `huggingface-cli login`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.data import normalize_text  # noqa: E402

SOURCES = {
    "wikipedia": {"path": "wikimedia/wikipedia", "name": "20231101.vi"},
    "culturax": {"path": "uonlp/CulturaX", "name": "vi"},
}

# Mỗi shard ~256MB văn bản: đủ nhỏ để mở lại bằng tay, đủ lớn để không sinh nghìn file.
SHARD_BYTES = 256 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), default="wikipedia")
    parser.add_argument("--target-gb", type=float, default=1.5)
    parser.add_argument("--out-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--min-chars", type=int, default=200, help="bỏ document quá ngắn")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Thiếu gói: pip install datasets", file=sys.stderr)
        return 1

    spec = SOURCES[args.source]
    target_bytes = int(args.target_gb * 1024**3)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # streaming=True: không tải nguyên bộ 68GB về đĩa rồi mới lọc.
    stream = load_dataset(spec["path"], spec["name"], split="train", streaming=True)

    total = shard_idx = shard_bytes = n_docs = 0
    fh = None
    try:
        for row in stream:
            text = normalize_text(row.get("text", ""))
            if len(text) < args.min_chars:
                continue
            if fh is None:
                path = args.out_dir / f"{args.source}_vi_{shard_idx:04d}.jsonl"
                fh = path.open("w", encoding="utf-8")
                print(f"-> {path.name}")
            import json

            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            size = len(text.encode("utf-8"))
            total += size
            shard_bytes += size
            n_docs += 1
            if n_docs % 10_000 == 0:
                print(f"   {n_docs:,} doc | {total / 1024**3:.2f} GB", flush=True)
            if shard_bytes >= SHARD_BYTES:
                fh.close()
                fh = None
                shard_bytes = 0
                shard_idx += 1
            if total >= target_bytes:
                break
    finally:
        if fh is not None:
            fh.close()

    print(f"Xong: {n_docs:,} document, {total / 1024**3:.2f} GB -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
