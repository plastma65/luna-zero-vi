"""Chạy generation final đã khóa sau khi hoàn tất tuning diagnostic.

Script cố ý không có tùy chọn đổi prompt, seed, profile hay số token. Muốn thay đổi
các thành phần đó phải mở khóa benchmark một cách có chủ ý trong source/config.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.checkpoint import CheckpointManager  # noqa: E402
from luna_zero.eval_sinh import (  # noqa: E402
    bao_dam_chua_chay_final,
    bao_dam_prompt_final_sach,
    bat_overlap_prompt,
    chay_benchmark,
    doc_suite,
    doc_suite_final,
    duong_dan_bao_cao_final,
    final_run_config,
    fingerprint_file,
    fingerprint_json,
    kiem_checkpoint_final,
    profile_mac_dinh,
    tom_tat,
)
from luna_zero.pack import kiem_tokenizer_checkpoint, tokenizer_fingerprint  # noqa: E402
from luna_zero.tokenizer import LunaTokenizer  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    p.add_argument("--tokenizer", type=Path, default=config.TOKENIZER_PATH)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="chỉ kiểm khóa/provenance/contamination, không sinh output final",
    )
    args = p.parse_args()

    import torch

    from luna_zero.config import chon_thiet_bi
    from luna_zero.model import LunaZeroGPT

    try:
        suite_meta, prompts = doc_suite_final(config.GEN_FINAL_PATH)
        _, diagnostic_prompts = doc_suite(config.GEN_DIAGNOSTIC_PATH)
        bat_overlap_prompt(prompts, diagnostic_prompts)
        n_docs_scan = bao_dam_prompt_final_sach(prompts, config.RAW_DIR)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Final suite không hợp lệ: {exc}", file=sys.stderr)
        return 2

    manager = CheckpointManager(args.checkpoint_dir)
    ckpt = manager.latest_path()
    if ckpt is None or not ckpt.exists():
        print(f"Không tìm thấy checkpoint trong {args.checkpoint_dir}", file=sys.stderr)
        return 1

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    step = int(blob["state"]["step"])
    ckpt_fp = fingerprint_file(ckpt)
    try:
        kiem_checkpoint_final(step, ckpt_fp)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    suite_fp = fingerprint_json(suite_meta)
    out_dir = config.ARTIFACT_DIR / "eval"
    out = duong_dan_bao_cao_final(out_dir, step, ckpt_fp, suite_fp)
    try:
        bao_dam_chua_chay_final(out)
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    run_spec = final_run_config()
    print(
        f"Final suite : {config.GEN_FINAL_PATH.name} | {len(prompts)} prompt\n"
        f"Checkpoint  : {ckpt.name} | bước {step:,}\n"
        f"Sampling    : {run_spec['profile']}\n"
        f"Seeds       : {run_spec['seeds']} | {run_spec['so_token']} token/mẫu\n"
        f"Contam scan : {n_docs_scan:,} doc raw corpus | exact prompt hit = 0"
    )
    if args.validate_only:
        print("VALIDATE-ONLY: khóa final hợp lệ; chưa sinh hoặc xem output final.")
        return 0

    kiem_tokenizer_checkpoint(blob, args.tokenizer)
    tok = LunaTokenizer.load(args.tokenizer)
    cfg = replace(config.MODEL, **blob.get("model_cfg", {}))
    if cfg.vocab_size != tok.vocab_size:
        print(
            f"Vocab lệch: checkpoint {cfg.vocab_size:,} vs tokenizer {tok.vocab_size:,}",
            file=sys.stderr,
        )
        return 1

    device = chon_thiet_bi(args.device)
    model = LunaZeroGPT(cfg).to(device)
    model.load_state_dict(blob["model"])
    model.eval()

    mau = chay_benchmark(
        model,
        tok,
        prompts,
        [profile_mac_dinh()],
        list(config.SINH_EVAL.seeds),
        config.SINH.so_token,
        device,
    )
    summary = tom_tat(mau)
    s = summary["mac_dinh"]
    print(
        f"Thiết bị    : {device}\n"
        f"Mẫu         : {len(mau):,}\n"
        f"distinct-2  : mean {s['distinct_2_mean']:.3f}\n"
        f"EOS         : {s['eos_rate']:.1%}\n"
        f"4-gram lặp  : max {s['max_4gram_repeat']}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "purpose": "final",
        "used_for_sampling_tuning": False,
        "eligible_for_final_claims": True,
        "limitations": suite_meta.get("limitations"),
        "suite_metadata": suite_meta,
        "contamination": {
            "raw_corpus_dir": str(config.RAW_DIR),
            "documents_scanned": n_docs_scan,
            "exact_prompt_hits": 0,
            "near_duplicate_or_paraphrase_checked": False,
        },
        "run": {
            **run_spec,
            "checkpoint": ckpt.name,
            "checkpoint_fingerprint": ckpt_fp,
            "prompt_suite_fingerprint": suite_fp,
            "tokenizer_fingerprint": tokenizer_fingerprint(args.tokenizer),
            "device": device,
        },
        "summary": summary,
        "samples": [x.to_dict() for x in mau],
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)
    print("Lưu ý       : không dùng output final này để tune lại sampling/checkpoint.")
    print(f"Đã lưu      : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
