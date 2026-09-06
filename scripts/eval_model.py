"""Final eval Luna Zero trên corpus held-out độc lập.

Tập này KHÔNG được là `val.bin`: validation đã dùng để chọn best.pt. Script bắt buộc
quét raw corpus cũ để chặn document eval từng xuất hiện trong cả train lẫn validation.

Ví dụ smoke trước:
    python scripts/eval_model.py --eval-corpus data/eval/final.jsonl --smoke

Chạy thật:
    python scripts/eval_model.py --eval-corpus data/eval/final.jsonl --best
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.checkpoint import CheckpointManager  # noqa: E402
from luna_zero.data import iter_corpus, iter_jsonl_texts  # noqa: E402
from luna_zero.eval import (  # noqa: E402
    bat_buoc_tap_eval_sach,
    do_loss_heldout,
    kiem_tra_tach_biet,
)
from luna_zero.pack import kiem_tokenizer_checkpoint, tokenizer_fingerprint  # noqa: E402
from luna_zero.tokenizer import LunaTokenizer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-corpus", type=Path, required=True)
    p.add_argument("--train-corpus-dir", type=Path, default=config.RAW_DIR)
    p.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    p.add_argument("--tokenizer", type=Path, default=config.TOKENIZER_PATH)
    p.add_argument("--device", default=None)
    p.add_argument("--best", action="store_true", help="đo best.pt thay vì checkpoint mới nhất")
    p.add_argument("--batch-size", type=int, default=config.EVAL.batch_size)
    p.add_argument("--smoke", action="store_true", help="chỉ 4 doc / 2 cửa sổ để kiểm pipeline")
    p.add_argument("--khong-luu", action="store_true")
    args = p.parse_args()

    if not args.eval_corpus.exists():
        print(f"Thiếu tập eval: {args.eval_corpus}", file=sys.stderr)
        return 1
    if not args.train_corpus_dir.exists():
        print(
            "Không có raw corpus cũ để chứng minh held-out tách biệt: " f"{args.train_corpus_dir}",
            file=sys.stderr,
        )
        return 1

    # Mở hai iterator riêng: kiểm contamination tiêu thụ iterator eval lần thứ nhất;
    # phép đo phải đọc lại từ đầu thay vì vô tình chỉ đo phần còn sót.
    kiem = kiem_tra_tach_biet(
        iter_jsonl_texts(args.eval_corpus),
        iter_corpus(args.train_corpus_dir),
    )
    try:
        bat_buoc_tap_eval_sach(kiem)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(
        f"Held-out   : {kiem.n_eval_docs:,} doc duy nhất; "
        f"đã quét {kiem.n_train_docs_quet:,} doc corpus cũ; trùng = 0"
    )

    import torch

    from luna_zero.config import chon_thiet_bi
    from luna_zero.model import LunaZeroGPT

    device = chon_thiet_bi(args.device)
    tok = LunaTokenizer.load(args.tokenizer)
    manager = CheckpointManager(args.checkpoint_dir)
    ckpt = args.checkpoint_dir / "best.pt" if args.best else manager.latest_path()
    if ckpt is None or not ckpt.exists():
        print(f"Không tìm thấy checkpoint trong {args.checkpoint_dir}", file=sys.stderr)
        return 1

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    kiem_tokenizer_checkpoint(blob, args.tokenizer)
    cfg = replace(config.MODEL, **blob.get("model_cfg", {}))
    if cfg.vocab_size != tok.vocab_size:
        print(
            f"Vocab lệch: checkpoint {cfg.vocab_size:,} vs tokenizer {tok.vocab_size:,}",
            file=sys.stderr,
        )
        return 1

    model = LunaZeroGPT(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    max_docs = config.EVAL.smoke_docs if args.smoke else None
    max_windows = config.EVAL.smoke_windows if args.smoke else None
    ket_qua = do_loss_heldout(
        model,
        tok,
        iter_jsonl_texts(args.eval_corpus),
        device=device,
        batch_size=args.batch_size,
        max_docs=max_docs,
        max_windows=max_windows,
    )

    step = int(blob["state"]["step"])
    print(f"Checkpoint : {ckpt.name} | bước {step:,} | thiết bị {device}")
    print(
        f"Đã đo      : {ket_qua.n_docs:,} doc / {ket_qua.n_windows:,} cửa sổ / "
        f"{ket_qua.n_tokens:,} token"
    )
    print(f"NLL        : {ket_qua.nll:.4f}")
    print(f"Perplexity : {ket_qua.perplexity:.3f}")

    if not args.khong_luu:
        out_dir = config.ARTIFACT_DIR / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"heldout_step_{step:07d}{'_smoke' if args.smoke else ''}.json"
        report = {
            "loai": "heldout_final_eval" if not args.smoke else "heldout_smoke",
            "checkpoint": str(ckpt),
            "step": step,
            "eval_corpus": str(args.eval_corpus),
            "train_corpus_dir_quet_contamination": str(args.train_corpus_dir),
            "tokenizer": str(args.tokenizer),
            "tokenizer_fingerprint": tokenizer_fingerprint(args.tokenizer),
            "kiem_tra_tach_biet": asdict(kiem),
            "ket_qua": asdict(ket_qua),
        }
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Đã lưu     : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
