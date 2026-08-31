"""Khử trùng lặp + token hoá corpus thành train.bin / val.bin.

python scripts/dong_goi.py
python scripts/dong_goi.py --max-docs 5000 --out-dir data/thu_pack   # chạy thử
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.data import iter_corpus  # noqa: E402
from luna_zero.pack import (  # noqa: E402
    kiem_tokenizer_khop,
    load_split,
    pack_documents,
)
from luna_zero.tokenizer import LunaTokenizer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--tokenizer", type=Path, default=config.TOKENIZER_PATH)
    parser.add_argument("--max-docs", type=int, default=None, help="chạy thử trên N doc")
    args = parser.parse_args()

    tok = LunaTokenizer.load(args.tokenizer)
    docs = iter_corpus(args.corpus_dir)
    if args.max_docs is not None:
        docs = itertools.islice(docs, args.max_docs)

    print(f"Tokenizer : {args.tokenizer} (vocab {tok.vocab_size:,})")
    print(f"Corpus    : {args.corpus_dir}")
    t0 = time.perf_counter()
    stats = pack_documents(tok, docs, args.out_dir, tokenizer_path=args.tokenizer)
    giay = time.perf_counter() - t0

    ty_le_trung = stats.n_docs_trung / stats.n_docs_vao if stats.n_docs_vao else 0.0
    print(f"\nXong sau {giay / 60:.1f} phút -> {args.out_dir}")
    print(f"Document vào   : {stats.n_docs_vao:,}")
    print(f"Trùng, đã bỏ   : {stats.n_docs_trung:,} ({ty_le_trung:.2%})")
    print(f"Token train    : {stats.n_train_tokens:,}")
    print(f"Token val      : {stats.n_val_tokens:,}")
    print(f"Tổng token     : {stats.n_tokens:,}")

    muc_tieu = config.TRAIN.target_tokens
    print(f"\nSo với mục tiêu {muc_tieu:,}: {stats.n_tokens / muc_tieu:.1%}")
    if stats.n_tokens < muc_tieu:
        thieu = muc_tieu - stats.n_tokens
        print(f"THIẾU {thieu:,} token. Tải thêm corpus bằng --resume rồi chạy lại.")

    # Kiểm chứng ngay: meta.json vừa ghi phải khớp với chính tokenizer vừa dùng.
    kiem_tokenizer_khop(args.out_dir, args.tokenizer)
    print("Vân tay tokenizer: khớp")

    # Kiểm chứng file ghi ra đọc lại được, không tin vào bộ đếm trong bộ nhớ.
    for split in ("train", "val"):
        arr = load_split(args.out_dir, split)
        print(f"{split}.bin: {arr.size:,} token, max id {int(arr.max()) if arr.size else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
