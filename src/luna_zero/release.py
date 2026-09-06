"""Đóng Chặng 4 và xuất package inference-only cho Hugging Face Hub."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from luna_zero import config
from luna_zero.eval import file_fingerprint
from luna_zero.eval_sinh import fingerprint_file, kiem_checkpoint_final


def _kiem_tokenizer_checkpoint(blob: dict[str, Any], path: Path) -> None:
    from luna_zero.pack import kiem_tokenizer_checkpoint

    kiem_tokenizer_checkpoint(blob, path)


def _tokenizer_fingerprint(path: Path) -> str:
    from luna_zero.pack import tokenizer_fingerprint

    return tokenizer_fingerprint(path)


def _doc_json(path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Không đọc được JSON artifact {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"Artifact {path} không phải JSON object.")
    return obj


def _tim_duy_nhat(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(
            f"Cần đúng 1 artifact khớp {pattern!r} trong {directory}, tìm thấy {len(paths)}."
        )
    return paths[0]


def tao_stage4_report(eval_dir: Path) -> dict[str, Any]:
    """Gom đúng hai phép đo final đã khóa; diagnostic không được nhập vào final metric."""
    gen_path = _tim_duy_nhat(
        eval_dir, f"generation_final_step_{config.SINH_EVAL.final_checkpoint_step:07d}_*.json"
    )
    loss_path = _tim_duy_nhat(
        eval_dir, f"heldout_human_final_step_{config.SINH_EVAL.final_checkpoint_step:07d}_*.json"
    )
    gen = _doc_json(gen_path)
    loss = _doc_json(loss_path)

    if gen.get("purpose") != "final" or not gen.get("eligible_for_final_claims"):
        raise ValueError("Generation artifact không được đánh dấu final/eligible.")
    if gen.get("used_for_sampling_tuning"):
        raise ValueError("Generation final đã bị dùng để tune sampling.")
    run = gen.get("run", {})
    if not isinstance(run, dict):
        raise ValueError("Generation artifact thiếu run metadata.")
    gen_contam = gen.get("contamination", {})
    if not isinstance(gen_contam, dict) or int(gen_contam.get("exact_prompt_hits", -1)) != 0:
        raise ValueError("Generation final chưa chứng minh exact prompt contamination = 0.")
    kiem_checkpoint_final(
        int(run.get("checkpoint_step", -1)),
        str(run.get("checkpoint_fingerprint", "")),
    )

    if loss.get("purpose") != "final_human_loss" or not loss.get("eligible_for_final_claims"):
        raise ValueError("Human-loss artifact không được đánh dấu final/eligible.")
    kiem_checkpoint_final(
        int(loss.get("checkpoint_step", -1)), str(loss.get("checkpoint_fingerprint", ""))
    )
    contam = loss.get("kiem_tra_tach_biet", {})
    if not isinstance(contam, dict) or int(contam.get("n_trung_train", -1)) != 0:
        raise ValueError("Human final loss chưa chứng minh exact train overlap = 0.")
    if loss.get("tokenizer_fingerprint") != run.get("tokenizer_fingerprint"):
        raise ValueError("Tokenizer fingerprint giữa hai final artifact không khớp.")

    ket_qua = loss.get("ket_qua", {})
    summary = gen.get("summary", {}).get(config.SINH_EVAL.final_profile_name)
    if not isinstance(ket_qua, dict) or not isinstance(summary, dict):
        raise ValueError("Final artifact thiếu metric bắt buộc.")

    for key in ("n_docs", "n_tokens", "nll", "perplexity"):
        if not isinstance(ket_qua.get(key), (int, float)):
            raise ValueError(f"Human final loss thiếu metric số {key!r}.")
    for key in (
        "n_samples",
        "eos_rate",
        "distinct_1_mean",
        "distinct_2_mean",
        "distinct_4_mean",
        "max_4gram_repeat",
    ):
        if not isinstance(summary.get(key), (int, float)):
            raise ValueError(f"Generation final thiếu metric số {key!r}.")

    return {
        "schema_version": config.RELEASE.schema_version,
        "stage": 4,
        "status": "complete",
        "model": {
            "name": "Luna Zero 110M",
            "trained_from_scratch": True,
            "model_config": asdict(config.MODEL),
            "target_train_tokens": config.TRAIN.target_tokens,
            "checkpoint_step": config.SINH_EVAL.final_checkpoint_step,
            "checkpoint_fingerprint": config.SINH_EVAL.final_checkpoint_fingerprint,
            "tokenizer_fingerprint": str(run.get("tokenizer_fingerprint", "")),
        },
        "final_human_loss": {
            "artifact": loss_path.name,
            "artifact_fingerprint": file_fingerprint(loss_path),
            "corpus_fingerprint": loss.get("eval_corpus_fingerprint"),
            "n_docs": ket_qua.get("n_docs"),
            "n_tokens": ket_qua.get("n_tokens"),
            "nll": ket_qua.get("nll"),
            "perplexity": ket_qua.get("perplexity"),
            "exact_train_overlap": contam.get("n_trung_train"),
            "train_docs_scanned": contam.get("n_train_docs_quet"),
            "limitations": loss.get("limitations"),
        },
        "final_generation": {
            "artifact": gen_path.name,
            "artifact_fingerprint": file_fingerprint(gen_path),
            "n_samples": summary.get("n_samples"),
            "distinct_1_mean": summary.get("distinct_1_mean"),
            "distinct_2_mean": summary.get("distinct_2_mean"),
            "distinct_4_mean": summary.get("distinct_4_mean"),
            "eos_rate": summary.get("eos_rate"),
            "max_4gram_repeat": summary.get("max_4gram_repeat"),
            "sampling": run.get("profile"),
            "limitations": gen.get("limitations"),
        },
        "measurement_policy": {
            "human_loss_locked_before_measurement": True,
            "generation_locked_before_measurement": True,
            "final_outputs_not_used_for_tuning": True,
            "near_duplicate_or_paraphrase_excluded": False,
        },
    }


def ghi_stage4_report(eval_dir: Path, output: Path) -> dict[str, Any]:
    report = tao_stage4_report(eval_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)
    return report


def render_stage4_markdown(report: dict[str, Any]) -> str:
    """Bản đọc cho người; số liệu đều lấy từ JSON final, không nhập lại bằng tay."""
    loss = report["final_human_loss"]
    gen = report["final_generation"]
    return f"""# Chặng 4 — Eval và sinh văn bản

Trạng thái: **hoàn tất**. Checkpoint phát hành: step
**{report['model']['checkpoint_step']:,}**.

## Final human-authored loss

- Dữ liệu: {loss['n_docs']} document / {loss['n_tokens']:,} token.
- NLL: **{loss['nll']:.4f}**.
- Perplexity: **{loss['perplexity']:.3f}**.
- Exact overlap với raw train corpus: **{loss['exact_train_overlap']}** sau khi quét
  {loss['train_docs_scanned']:,} document.

## Final generation

- {gen['n_samples']} mẫu, cấu hình sampling đã khóa trước khi chạy final.
- distinct-2 mean: **{gen['distinct_2_mean']:.3f}**.
- distinct-4 mean: **{gen['distinct_4_mean']:.3f}**.
- 4-gram lặp tối đa: **{gen['max_4gram_repeat']}**.
- EOS rate: **{gen['eos_rate']:.1%}**.

Các distinct-n chỉ mô tả repetition; không phải điểm factuality hay chất lượng nội dung.

## Chính sách đo

Human-loss corpus và generation suite đều được khóa trước khi đo. Output final không
được dùng để tune sampling hoặc chọn lại checkpoint. Exact contamination đã được kiểm
tra; near-duplicate/paraphrase **chưa được loại trừ** và phải được giữ như một giới hạn
của phép đo.
"""


def _state_dict_inference(blob: dict[str, Any]) -> dict[str, Any]:
    state = blob.get("model")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint thiếu model state_dict.")
    # wte và lm_head bị tied. Safetensors không cần lưu cùng tensor hai lần;
    # loader dựng model đã tie sẵn rồi nạp wte.weight là đủ.
    out = {}
    for key, tensor in state.items():
        if key == "lm_head.weight":
            continue
        out[key] = tensor.detach().cpu().contiguous()
    if "transformer.wte.weight" not in out:
        raise ValueError("Checkpoint thiếu transformer.wte.weight.")
    return out


def _model_card(report: dict[str, Any]) -> str:
    loss = report["final_human_loss"]
    gen = report["final_generation"]
    return f"""---
language:
- vi
pipeline_tag: text-generation
library_name: pytorch
---

# Luna Zero 110M

Luna Zero là mô hình ngôn ngữ decoder-only tiếng Việt khoảng 110M tham số, **train hoàn
toàn từ số 0**; không fine-tune và không nạp trọng số pretrained của mô hình khác.

## Kiến trúc

- 12 Transformer block, d_model 768, 12 attention head.
- Context tối đa 1024 token.
- Byte-level BPE 32.000 token, NFC ở data layer, không có UNK.
- Input embedding và LM head dùng chung trọng số.

## Train

Mục tiêu train là {config.TRAIN.target_tokens:,} token. Bản phát hành này dùng checkpoint
step {config.SINH_EVAL.final_checkpoint_step:,}, được khóa trước final eval.

## Final eval

Human-authored held-out: {loss['n_docs']} document / {loss['n_tokens']:,} token,
NLL **{loss['nll']:.4f}**, perplexity **{loss['perplexity']:.3f}**. Exact document overlap
với raw train corpus bằng 0 sau khi quét {loss['train_docs_scanned']:,} document.

Final generation: {gen['n_samples']} mẫu, distinct-2 mean **{gen['distinct_2_mean']:.3f}**,
distinct-4 mean **{gen['distinct_4_mean']:.3f}**, 4-gram lặp tối đa
**{gen['max_4gram_repeat']}**. Các metric này mô tả repetition, **không phải** điểm
factuality hay chất lượng nội dung.

## Giới hạn

Model ở quy mô này có thể viết tiếng Việt khá trôi nhưng thường bịa dữ kiện, mất mạch
ngữ nghĩa và không nên dùng như nguồn thông tin đáng tin cậy. Kiểm contamination hiện
chứng minh exact document/prompt non-overlap; chưa loại được near-duplicate hoặc
paraphrase.

Final eval đã được khóa trước khi đo và không dùng output final để tune sampling hoặc
chọn lại checkpoint. Xem `stage4_report.json` và `generation_final.json` để có provenance
chi tiết.

## Sampling mặc định

```json
{json.dumps(asdict(config.SINH), ensure_ascii=False, indent=2)}
```

## Load

Package này dùng kiến trúc custom của repo Luna Zero, không giả vờ tương thích trực tiếp
với `transformers.AutoModel`. Dùng code Luna Zero để dựng `LunaZeroGPT`, nạp
`model.safetensors`, và dùng tokenizer đi kèm. `scripts/kiem_hf_package.py` là smoke test
CPU cho đúng luồng load đó.
"""


def xuat_hf_package(
    checkpoint: Path,
    report_path: Path,
    generation_artifact: Path,
    output_dir: Path,
) -> Path:
    """Xuất trọng số inference-only sau khi xác minh checkpoint/tokenizer/report final."""
    import torch
    from safetensors.torch import save_file

    report = _doc_json(report_path)
    if report.get("stage") != 4 or report.get("status") != "complete":
        raise ValueError("Stage 4 report chưa complete.")

    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step = int(blob.get("state", {}).get("step", -1))
    ckpt_fp = fingerprint_file(checkpoint)
    kiem_checkpoint_final(step, ckpt_fp)
    _kiem_tokenizer_checkpoint(blob, config.TOKENIZER_PATH)

    cfg = replace(config.MODEL, **blob.get("model_cfg", {}))
    package = output_dir / config.RELEASE.package_dir_name
    if package.exists():
        shutil.rmtree(package)
    package.mkdir(parents=True)

    weights = package / config.RELEASE.weights_name
    save_file(
        _state_dict_inference(blob),
        str(weights),
        metadata={
            "format": "luna_zero",
            "checkpoint_step": str(step),
            "checkpoint_fingerprint": ckpt_fp,
            "tokenizer_fingerprint": _tokenizer_fingerprint(config.TOKENIZER_PATH),
            "tied_weight": "lm_head.weight=transformer.wte.weight",
        },
    )

    model_cfg = {
        "architectures": ["LunaZeroGPT"],
        "model_type": "luna_zero",
        **asdict(cfg),
        "bos_token_id": config.BOS_ID,
        "eos_token_id": config.EOS_ID,
        "pad_token_id": config.PAD_ID,
        "tie_word_embeddings": True,
        "trained_from_scratch": True,
        "checkpoint_step": step,
        "checkpoint_fingerprint": ckpt_fp,
        "tokenizer_fingerprint": _tokenizer_fingerprint(config.TOKENIZER_PATH),
    }
    (package / config.RELEASE.model_config_name).write_text(
        json.dumps(model_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (package / config.RELEASE.generation_config_name).write_text(
        json.dumps(asdict(config.SINH), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(config.TOKENIZER_PATH, package / config.RELEASE.tokenizer_name)
    meta = config.TOKENIZER_PATH.with_suffix(".meta.json")
    if meta.exists():
        shutil.copy2(meta, package / config.RELEASE.tokenizer_meta_name)
    shutil.copy2(report_path, package / config.RELEASE.stage4_report_name)
    if config.STAGE4_REPORT_MD_PATH.exists():
        shutil.copy2(
            config.STAGE4_REPORT_MD_PATH,
            package / config.RELEASE.stage4_report_md_name,
        )
    shutil.copy2(generation_artifact, package / config.RELEASE.final_generation_name)
    (package / config.RELEASE.model_card_name).write_text(_model_card(report), encoding="utf-8")
    return package


def load_hf_package_cpu(package: Path) -> Any:
    """Smoke/load-roundtrip thật từ safetensors, không dựa vào checkpoint train."""
    import torch
    from safetensors.torch import load_file

    from luna_zero.config import ModelConfig
    from luna_zero.model import LunaZeroGPT

    cfg_raw = _doc_json(package / config.RELEASE.model_config_name)
    cfg = ModelConfig(**{k: cfg_raw[k] for k in asdict(config.MODEL)})
    model = LunaZeroGPT(cfg).to("cpu")
    state = load_file(str(package / config.RELEASE.weights_name), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["lm_head.weight"] or unexpected:
        raise ValueError(f"State dict package lệch: missing={missing}, unexpected={unexpected}")
    if model.lm_head.weight.data_ptr() != model.transformer["wte"].weight.data_ptr():
        raise ValueError("Weight tying bị mất sau khi load package.")
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[config.BOS_ID, config.EOS_ID]], dtype=torch.long)
        y = model(x).logits
    if tuple(y.shape) != (1, 2, cfg.vocab_size):
        raise ValueError(f"Forward smoke sai shape: {tuple(y.shape)}")
    return model
