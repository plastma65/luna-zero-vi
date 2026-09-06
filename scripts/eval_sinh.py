"""Benchmark generation diagnostic có seed, raw output và provenance.

Ví dụ:
  python scripts/eval_sinh.py --checkpoint-dir C:\\Luna_checkpoints --smoke
  python scripts/eval_sinh.py --checkpoint-dir C:\\Luna_checkpoints

Suite này ĐƯỢC PHÉP dùng để tune sampling, vì vậy kết quả không được gọi là
final benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.checkpoint import CheckpointManager  # noqa: E402
from luna_zero.eval_sinh import (  # noqa: E402
    chay_benchmark,
    doc_suite,
    fingerprint_file,
    profiles_mac_dinh,
    tom_tat,
)
from luna_zero.pack import kiem_tokenizer_checkpoint, tokenizer_fingerprint  # noqa: E402
from luna_zero.tokenizer import LunaTokenizer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    p.add_argument("--tokenizer", type=Path, default=config.TOKENIZER_PATH)
    p.add_argument("--prompt-suite", type=Path, default=config.GEN_DIAGNOSTIC_PATH)
    p.add_argument("--device", default=None)
    p.add_argument("--best", action="store_true", help="dùng best.pt thay vì bản mới nhất")
    p.add_argument("--so-token", type=int, default=config.SINH.so_token)
    p.add_argument("--seed", type=int, action="append", default=None, help="lặp được")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--khong-luu", action="store_true")
    args = p.parse_args()

    import torch

    from luna_zero.config import chon_thiet_bi
    from luna_zero.model import LunaZeroGPT

    try:
        suite_meta, prompts = doc_suite(args.prompt_suite)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Prompt suite không hợp lệ: {exc}", file=sys.stderr)
        return 2

    seeds = list(args.seed) if args.seed is not None else list(config.SINH_EVAL.seeds)
    profiles = profiles_mac_dinh()
    so_token = args.so_token
    if args.smoke:
        prompts = prompts[: config.SINH_EVAL.smoke_prompts]
        seeds = seeds[: config.SINH_EVAL.smoke_seeds]
        so_token = min(so_token, config.SINH_EVAL.smoke_tokens)

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
    step = int(blob["state"]["step"])

    mau = chay_benchmark(model, tok, prompts, profiles, seeds, so_token, device)
    summary = tom_tat(mau)

    print(
        f"Checkpoint : {ckpt.name} | bước {step:,} | thiết bị {device}\n"
        f"Suite      : {args.prompt_suite.name} | {len(prompts)} prompt | "
        f"{len(seeds)} seed | {len(profiles)} profile\n"
        f"Mẫu        : {len(mau):,} | tối đa {so_token} token/mẫu"
    )
    for ten, s in summary.items():
        print(
            f"{ten:15s}: distinct-2 mean {s['distinct_2_mean']:.3f} | "
            f"EOS {s['eos_rate']:.1%} | 4-gram lặp max {s['max_4gram_repeat']}"
        )
    print("Lưu ý      : diagnostic/tuning only; không phải final benchmark.")

    if not args.khong_luu:
        out_dir = config.ARTIFACT_DIR / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = {
            "step": step,
            "checkpoint": ckpt.name,
            "checkpoint_fingerprint": fingerprint_file(ckpt),
            "prompt_suite_fingerprint": fingerprint_file(args.prompt_suite),
            "tokenizer_fingerprint": tokenizer_fingerprint(args.tokenizer),
            "seeds": seeds,
            "profiles": [asdict(x) for x in profiles],
            "so_token": so_token,
            "smoke": args.smoke,
        }
        spec_bytes = json.dumps(spec, ensure_ascii=False, sort_keys=True).encode("utf-8")
        run_id = hashlib.blake2b(spec_bytes, digest_size=6).hexdigest()
        suffix = "_smoke" if args.smoke else ""
        out = out_dir / f"generation_diagnostic_step_{step:07d}_{run_id}{suffix}.json"
        report = {
            "schema_version": 1,
            "purpose": "diagnostic",
            "used_for_sampling_tuning": True,
            "eligible_for_final_claims": False,
            "suite_metadata": suite_meta,
            "run": spec,
            "summary": summary,
            "samples": [x.to_dict() for x in mau],
        }
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(out)
        print(f"Đã lưu     : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
