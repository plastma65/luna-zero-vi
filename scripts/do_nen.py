"""Đo tỷ lệ nén của tokenizer (ký tự/token) và so với tokenizer mức ký tự.

python scripts/do_nen.py
python scripts/do_nen.py --tokenizer artifacts/tokenizer/smoke_bpe.json --sample-docs 200
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.data import iter_corpus  # noqa: E402
from luna_zero.tokenizer import (  # noqa: E402
    LunaTokenizer,
    char_level_baseline,
    doc_train_bytes,
    measure_compression,
    meta_path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=config.TOKENIZER_PATH)
    parser.add_argument("--corpus-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--sample-docs", type=int, default=5_000)
    parser.add_argument(
        "--skip-bytes",
        type=int,
        default=None,
        help="bỏ qua N byte đầu corpus trước khi lấy mẫu. Mặc định đọc từ metadata "
        "của tokenizer, tức đúng phần nó đã học.",
    )
    args = parser.parse_args()

    tok = LunaTokenizer.load(args.tokenizer)
    # Bỏ qua đúng phần corpus đã dùng để train tokenizer rồi mới lấy mẫu.
    # Bài học Luna cũ: bộ eval trùng dữ liệu train thì phép đo chỉ đo trí nhớ.
    # Vừa tránh ô nhiễm, vừa không phải nạp cả corpus vào RAM để lấy đuôi.
    if args.skip_bytes is not None:
        bo_qua_bytes, nguon = args.skip_bytes, "tham số --skip-bytes"
    elif meta_path(args.tokenizer).exists():
        bo_qua_bytes, nguon = doc_train_bytes(args.tokenizer), "metadata tokenizer"
    else:
        bo_qua_bytes = config.TOKENIZER.train_bytes
        nguon = "config (CẢNH BÁO: không có metadata, có thể đo trúng dữ liệu train)"
    print(f"Bỏ qua      : {bo_qua_bytes / 1024**2:,.0f} MB đầu corpus [{nguon}]")

    stream = iter_corpus(args.corpus_dir)
    skipped = 0
    for text in stream:
        skipped += len(text.encode("utf-8"))
        if skipped >= bo_qua_bytes:
            break
    docs = list(itertools.islice(stream, args.sample_docs))
    if not docs:
        # Corpus nhỏ hơn phần train (ví dụ khi chạy smoke): đành đo trên đuôi corpus.
        docs = list(deque(iter_corpus(args.corpus_dir), maxlen=args.sample_docs))
    if not docs:
        print(f"Corpus rỗng: {args.corpus_dir}", file=sys.stderr)
        return 1

    stats = measure_compression(tok, docs)
    base = char_level_baseline(docs)
    target = config.TOKENIZER.min_chars_per_token

    print(f"Tokenizer   : {args.tokenizer}")
    print(f"Vocab       : {tok.vocab_size:,}")
    print(f"Mẫu đo      : {stats.n_docs:,} doc / {stats.n_chars:,} ký tự")
    print("-" * 46)
    print(f"{'BPE Luna Zero':<22}{stats.chars_per_token:>8.3f} ký tự/token")
    print(f"{'Mức ký tự (mốc)':<22}{base.chars_per_token:>8.3f} ký tự/token")
    print(f"{'Nén hơn mốc':<22}{stats.chars_per_token / base.chars_per_token:>8.2f}x")
    print("-" * 46)
    ok = stats.chars_per_token >= target
    print(f"Ngưỡng tối thiểu {target}: {'ĐẠT' if ok else 'KHÔNG ĐẠT'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
