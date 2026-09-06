"""Đo final NLL/PPL đúng một lần trên corpus human-authored đã khóa."""

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
    doc_meta_human_final,
    file_fingerprint,
    kiem_tra_tach_biet,
)
from luna_zero.eval_sinh import fingerprint_file, kiem_checkpoint_final  # noqa: E402
from luna_zero.pack import kiem_tokenizer_checkpoint, tokenizer_fingerprint  # noqa: E402
from luna_zero.tokenizer import LunaTokenizer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=config.EVAL.batch_size)
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args()

    eval_path = config.FINAL_HUMAN_EVAL_PATH
    meta_path = config.FINAL_HUMAN_META_PATH
    try:
        meta = doc_meta_human_final(meta_path, eval_path)
        kiem = kiem_tra_tach_biet(iter_jsonl_texts(eval_path), iter_corpus(config.RAW_DIR))
        bat_buoc_tap_eval_sach(kiem)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    manager = CheckpointManager(args.checkpoint_dir)
    ckpt = manager.latest_path()
    if ckpt is None or not ckpt.exists():
        print(f"Không tìm thấy checkpoint trong {args.checkpoint_dir}", file=sys.stderr)
        return 1

    import torch

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    step = int(blob["state"]["step"])
    ckpt_fp = fingerprint_file(ckpt)
    try:
        kiem_checkpoint_final(step, ckpt_fp)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Human final : {meta['n_docs']:,} doc | fingerprint {meta['output_fingerprint']}")
    print(f"Checkpoint  : {ckpt.name} | bước {step:,}")
    print(f"Contam scan : {kiem.n_train_docs_quet:,} doc raw | exact overlap = 0")
    if args.validate_only:
        print("VALIDATE-ONLY: khóa final loss hợp lệ; chưa tính NLL/PPL.")
        return 0

    from luna_zero.config import chon_thiet_bi
    from luna_zero.model import LunaZeroGPT

    device = chon_thiet_bi(args.device)
    tok = LunaTokenizer.load(config.TOKENIZER_PATH)
    kiem_tokenizer_checkpoint(blob, config.TOKENIZER_PATH)
    cfg = replace(config.MODEL, **blob.get("model_cfg", {}))
    model = LunaZeroGPT(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    out_dir = config.ARTIFACT_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = file_fingerprint(meta_path)[:12]
    out = out_dir / f"heldout_human_final_step_{step:07d}_{run_id}.json"
    if out.exists():
        print(f"Final loss đã có artifact {out}; từ chối chạy lại cùng metadata.", file=sys.stderr)
        return 2

    ket_qua = do_loss_heldout(
        model,
        tok,
        iter_jsonl_texts(eval_path),
        device=device,
        batch_size=args.batch_size,
    )
    report = {
        "schema_version": 1,
        "purpose": "final_human_loss",
        "eligible_for_final_claims": True,
        "checkpoint": str(ckpt),
        "checkpoint_step": step,
        "checkpoint_fingerprint": ckpt_fp,
        "tokenizer_fingerprint": tokenizer_fingerprint(config.TOKENIZER_PATH),
        "eval_corpus_fingerprint": meta["output_fingerprint"],
        "eval_metadata_fingerprint": file_fingerprint(meta_path),
        "provenance": meta["provenance"],
        "limitations": meta["limitations"],
        "kiem_tra_tach_biet": asdict(kiem),
        "ket_qua": asdict(ket_qua),
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Thiết bị     : {device}")
    print(f"Đã đo        : {ket_qua.n_docs:,} doc / {ket_qua.n_tokens:,} token")
    print(f"NLL          : {ket_qua.nll:.4f}")
    print(f"Perplexity   : {ket_qua.perplexity:.3f}")
    print(f"Đã lưu       : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
