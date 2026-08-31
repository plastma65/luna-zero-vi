"""Train tokenizer BPE 32k trên corpus tiếng Việt.

python scripts/train_tokenizer.py                  # thật, ~500MB text
python scripts/train_tokenizer.py --smoke          # 2MB, vocab 2k, chạy vài giây
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.data import iter_corpus  # noqa: E402
from luna_zero.tokenizer import train_tokenizer  # noqa: E402

SMOKE_BYTES = 2 * 1024 * 1024
SMOKE_VOCAB = 2_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out", type=Path, default=config.TOKENIZER_PATH)
    # default=None để phân biệt "người dùng không truyền" với "truyền đúng giá trị mặc định":
    # --smoke chỉ được đè lên thứ người dùng KHÔNG chỉ định.
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--train-bytes", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="chạy tí hon để kiểm pipeline trước khi đốt 20 phút CPU",
    )
    args = parser.parse_args()

    if args.smoke:
        args.train_bytes = args.train_bytes or SMOKE_BYTES
        args.vocab_size = args.vocab_size or SMOKE_VOCAB
        args.out = args.out.with_name("smoke_bpe.json")
    else:
        args.train_bytes = args.train_bytes or config.TOKENIZER.train_bytes
        args.vocab_size = args.vocab_size or config.TOKENIZER.vocab_size

    if not any(args.corpus_dir.glob("*.jsonl")):
        print(
            f"Không thấy .jsonl nào trong {args.corpus_dir}.\n"
            "Chạy trước: python scripts/download_corpus.py --source wikipedia",
            file=sys.stderr,
        )
        return 1

    print(f"Corpus  : {args.corpus_dir} (đọc tối đa {args.train_bytes / 1024**2:.0f} MB)")
    print(f"Vocab   : {args.vocab_size:,}")
    t0 = time.perf_counter()
    tok = train_tokenizer(
        iter_corpus(args.corpus_dir, max_bytes=args.train_bytes),
        output_path=args.out,
        vocab_size=args.vocab_size,
        train_bytes=args.train_bytes,
    )
    print(f"Xong sau {time.perf_counter() - t0:.1f}s -> {args.out}")
    print(f"Vocab thực tế: {tok.get_vocab_size():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
