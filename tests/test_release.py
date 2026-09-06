from __future__ import annotations

import json
from pathlib import Path

import pytest

from luna_zero import config
from luna_zero.release import (
    _state_dict_inference,
    load_hf_package_cpu,
    tao_stage4_report,
    xuat_hf_package,
)


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _final_artifacts(root: Path) -> None:
    gen = {
        "purpose": "final",
        "used_for_sampling_tuning": False,
        "eligible_for_final_claims": True,
        "limitations": "exact only",
        "contamination": {"exact_prompt_hits": 0},
        "run": {
            "checkpoint_step": config.SINH_EVAL.final_checkpoint_step,
            "checkpoint_fingerprint": config.SINH_EVAL.final_checkpoint_fingerprint,
            "tokenizer_fingerprint": "tok-fp",
            "profile": {"ten": "mac_dinh"},
        },
        "summary": {
            config.SINH_EVAL.final_profile_name: {
                "n_samples": 30,
                "eos_rate": 0.1,
                "distinct_1_mean": 0.8,
                "distinct_2_mean": 0.95,
                "distinct_4_mean": 0.99,
                "max_4gram_repeat": 2,
            }
        },
    }
    loss = {
        "purpose": "final_human_loss",
        "eligible_for_final_claims": True,
        "checkpoint_step": config.SINH_EVAL.final_checkpoint_step,
        "checkpoint_fingerprint": config.SINH_EVAL.final_checkpoint_fingerprint,
        "eval_corpus_fingerprint": "corpus-fp",
        "tokenizer_fingerprint": "tok-fp",
        "limitations": "near duplicate not checked",
        "kiem_tra_tach_biet": {"n_trung_train": 0, "n_train_docs_quet": 123},
        "ket_qua": {
            "n_docs": 100,
            "n_tokens": 17580,
            "nll": 3.2195,
            "perplexity": 25.016,
        },
    }
    _write_json(
        root / f"generation_final_step_{config.SINH_EVAL.final_checkpoint_step:07d}_abc.json",
        gen,
    )
    _write_json(
        root / f"heldout_human_final_step_{config.SINH_EVAL.final_checkpoint_step:07d}_def.json",
        loss,
    )


def test_stage4_report_chi_nhan_final_artifacts(tmp_path: Path) -> None:
    _final_artifacts(tmp_path)
    report = tao_stage4_report(tmp_path)
    assert report["status"] == "complete"
    assert report["final_human_loss"]["nll"] == pytest.approx(3.2195)
    assert report["final_generation"]["distinct_2_mean"] == pytest.approx(0.95)
    assert report["measurement_policy"]["final_outputs_not_used_for_tuning"] is True


def test_stage4_report_tu_choi_generation_da_dung_tune(tmp_path: Path) -> None:
    _final_artifacts(tmp_path)
    path = next(tmp_path.glob("generation_final*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    data["used_for_sampling_tuning"] = True
    _write_json(path, data)
    with pytest.raises(ValueError, match="tune sampling"):
        tao_stage4_report(tmp_path)


def test_state_dict_inference_bo_ban_sao_lm_head() -> None:
    import torch

    weight = torch.randn(4, 3)
    state = {
        "transformer.wte.weight": weight,
        "lm_head.weight": weight,
        "transformer.ln_f.weight": torch.ones(3),
    }
    out = _state_dict_inference({"model": state})
    assert "lm_head.weight" not in out
    assert "transformer.wte.weight" in out


def test_export_va_load_roundtrip_cpu_tu_safetensors(tmp_path: Path, monkeypatch) -> None:
    import torch

    import luna_zero.release as release
    from luna_zero.config import ModelConfig
    from luna_zero.model import LunaZeroGPT

    tiny = ModelConfig(n_layer=1, n_head=1, d_model=8, block_size=8, vocab_size=16)
    model = LunaZeroGPT(tiny)
    ckpt = tmp_path / "ckpt.pt"
    torch.save(
        {
            "state": {"step": config.SINH_EVAL.final_checkpoint_step},
            "model": model.state_dict(),
            "model_cfg": tiny.__dict__,
            "tokenizer_fingerprint": "tok-fp",
        },
        ckpt,
    )
    tok = tmp_path / "tok.json"
    tok.write_text("{}", encoding="utf-8")
    tok.with_suffix(".meta.json").write_text("{}", encoding="utf-8")
    report = tmp_path / "stage4_report.json"
    _write_json(
        report,
        {
            "stage": 4,
            "status": "complete",
            "final_human_loss": {
                "n_docs": 100,
                "n_tokens": 17580,
                "nll": 3.2195,
                "perplexity": 25.016,
                "train_docs_scanned": 123,
            },
            "final_generation": {
                "n_samples": 30,
                "distinct_2_mean": 0.95,
                "distinct_4_mean": 0.99,
                "max_4gram_repeat": 2,
            },
        },
    )
    gen = tmp_path / "generation.json"
    _write_json(gen, {"purpose": "final"})

    monkeypatch.setattr(release.config, "TOKENIZER_PATH", tok)
    monkeypatch.setattr(
        release,
        "fingerprint_file",
        lambda _: config.SINH_EVAL.final_checkpoint_fingerprint,
    )
    monkeypatch.setattr(release, "kiem_checkpoint_final", lambda step, fp: None)
    monkeypatch.setattr(release, "_kiem_tokenizer_checkpoint", lambda blob, path: None)
    monkeypatch.setattr(release, "_tokenizer_fingerprint", lambda path: "tok-fp")

    package = xuat_hf_package(ckpt, report, gen, tmp_path / "release")
    loaded = load_hf_package_cpu(package)
    assert loaded.cfg == tiny
    assert loaded.lm_head.weight.data_ptr() == loaded.transformer["wte"].weight.data_ptr()
    assert (package / config.RELEASE.weights_name).stat().st_size > 0
