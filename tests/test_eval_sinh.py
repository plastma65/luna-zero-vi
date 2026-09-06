from __future__ import annotations

import json
from pathlib import Path

import pytest

from luna_zero import config
from luna_zero.do_sinh import DoLap
from luna_zero.eval_sinh import (
    MauSinh,
    ProfileSinh,
    PromptDiagnostic,
    bao_dam_chua_chay_final,
    bao_dam_prompt_final_sach,
    bat_overlap_prompt,
    chay_benchmark,
    doc_suite,
    doc_suite_final,
    duong_dan_bao_cao_final,
    final_run_fingerprint,
    fingerprint_json,
    kiem_checkpoint_final,
    profile_mac_dinh,
    profiles_mac_dinh,
    tom_tat,
)


def _ghi_suite(path: Path, **override: object) -> None:
    data: dict[str, object] = {
        "purpose": "diagnostic",
        "used_for_sampling_tuning": True,
        "eligible_for_final_claims": False,
        "prompts": [
            {"id": "a", "nhom": "x", "text": "Một câu mở đầu"},
            {"id": "b", "nhom": "y", "text": "Một câu khác"},
        ],
    }
    data.update(override)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _ghi_final(path: Path, **override: object) -> dict[str, object]:
    data: dict[str, object] = {
        "purpose": "final",
        "used_for_sampling_tuning": False,
        "eligible_for_final_claims": True,
        "run_config_fingerprint": final_run_fingerprint(),
        "prompts": [
            {"id": "a", "nhom": "x", "text": "Một final prompt đủ riêng biệt"},
            {"id": "b", "nhom": "y", "text": "Một final prompt khác hoàn toàn"},
        ],
    }
    data.update(override)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def test_suite_diagnostic_bat_buoc_tu_choi_final_claim(tmp_path: Path) -> None:
    p = tmp_path / "suite.json"
    _ghi_suite(p, eligible_for_final_claims=True)
    with pytest.raises(ValueError, match="eligible_for_final_claims"):
        doc_suite(p)


def test_suite_bat_trung_prompt_text(tmp_path: Path) -> None:
    p = tmp_path / "suite.json"
    _ghi_suite(
        p,
        prompts=[
            {"id": "a", "nhom": "x", "text": "trùng"},
            {"id": "b", "nhom": "y", "text": "trùng"},
        ],
    )
    with pytest.raises(ValueError, match="Trùng prompt text"):
        doc_suite(p)


def test_profiles_lay_gia_tri_tu_config() -> None:
    profiles = {p.ten: p for p in profiles_mac_dinh()}
    assert set(profiles) == {"mac_dinh", "sample_lich_su"}
    assert profiles["mac_dinh"] == profile_mac_dinh()
    assert profiles["mac_dinh"].top_p is not None
    assert profiles["sample_lich_su"].top_k is not None


def test_final_suite_that_dang_duoc_khoa() -> None:
    raw, prompts = doc_suite_final(config.GEN_FINAL_PATH)
    assert raw["purpose"] == "final"
    assert len(prompts) > 1


def test_final_suite_tu_choi_khi_run_config_da_doi(tmp_path: Path) -> None:
    p = tmp_path / "final.json"
    raw = _ghi_final(p, run_config_fingerprint="sai")
    with pytest.raises(ValueError, match="Cấu hình generation final đã đổi"):
        doc_suite_final(p, expected_suite_fingerprint=fingerprint_json(raw))


def test_final_suite_tu_choi_khi_prompt_bi_sua_sau_khi_khoa(tmp_path: Path) -> None:
    p = tmp_path / "final.json"
    raw = _ghi_final(p)
    expected = fingerprint_json(raw)
    raw["prompts"] = [{"id": "x", "nhom": "x", "text": "prompt bị đổi"}]
    p.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="prompt suite đã đổi"):
        doc_suite_final(p, expected_suite_fingerprint=expected)


def test_final_prompt_khong_duoc_trung_diagnostic() -> None:
    final = [PromptDiagnostic("f", "x", "Câu đã dùng để tune")]
    diagnostic = [PromptDiagnostic("d", "x", "Câu đã dùng để tune")]
    with pytest.raises(ValueError, match="trùng diagnostic"):
        bat_overlap_prompt(final, diagnostic)


def test_final_prompt_bi_bat_neu_xuat_hien_trong_corpus_train(tmp_path: Path) -> None:
    corpus = tmp_path / "raw"
    corpus.mkdir()
    (corpus / "a.jsonl").write_text(
        json.dumps({"text": "Phần đầu. Câu final rất riêng. Phần cuối."}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    prompts = [PromptDiagnostic("f", "x", "Câu final rất riêng")]
    with pytest.raises(ValueError, match="xuất hiện trong corpus train"):
        bao_dam_prompt_final_sach(prompts, corpus)


def test_final_prompt_sach_tra_ve_so_doc_da_quet(tmp_path: Path) -> None:
    corpus = tmp_path / "raw"
    corpus.mkdir()
    (corpus / "a.jsonl").write_text(
        json.dumps({"text": "Văn bản hoàn toàn khác"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    prompts = [PromptDiagnostic("f", "x", "Câu final rất riêng")]
    assert bao_dam_prompt_final_sach(prompts, corpus) == 1


def test_final_tu_choi_checkpoint_khac_object_da_chon() -> None:
    kiem_checkpoint_final(
        config.SINH_EVAL.final_checkpoint_step,
        config.SINH_EVAL.final_checkpoint_fingerprint,
    )
    with pytest.raises(ValueError, match="sai fingerprint"):
        kiem_checkpoint_final(config.SINH_EVAL.final_checkpoint_step, "khac")


def test_final_tu_choi_raw_corpus_rong(tmp_path: Path) -> None:
    corpus = tmp_path / "raw"
    corpus.mkdir()
    prompts = [PromptDiagnostic("f", "x", "Câu final rất riêng")]
    with pytest.raises(ValueError, match="Không quét được document"):
        bao_dam_prompt_final_sach(prompts, corpus)


def test_final_artifact_deterministic_va_tu_choi_chay_lai(tmp_path: Path) -> None:
    a = duong_dan_bao_cao_final(tmp_path, 7, "abc", "def")
    b = duong_dan_bao_cao_final(tmp_path, 7, "abc", "def")
    assert a == b
    bao_dam_chua_chay_final(a)
    a.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="không chạy lại"):
        bao_dam_chua_chay_final(a)


def test_tom_tat_chi_bao_metric_tho_khong_gan_nhan_chat_luong() -> None:
    mau = [
        MauSinh(
            prompt_id="a",
            prompt="p",
            nhom="x",
            seed=1,
            profile="mac_dinh",
            output="a b c",
            eos=False,
            n_token_sinh=3,
            lap=DoLap(1.0, 1.0, 0.0, 0, 3),
        ),
        MauSinh(
            prompt_id="a",
            prompt="p",
            nhom="x",
            seed=2,
            profile="mac_dinh",
            output="a a a",
            eos=True,
            n_token_sinh=3,
            lap=DoLap(1 / 3, 0.5, 0.0, 0, 3),
        ),
    ]
    s = tom_tat(mau)["mac_dinh"]
    assert s["n_samples"] == 2
    assert s["eos_rate"] == 0.5
    assert s["distinct_2_mean"] == 0.75
    assert not any("quality" in k or "tu_nhien" in k for k in s)


def test_chay_benchmark_cung_seed_tai_lap() -> None:
    import torch

    class TokGia:
        def encode(self, text: str) -> list[int]:
            return [7, 8]

        def decode(self, ids: list[int]) -> str:
            return " ".join(str(x) for x in ids)

    class ModelGia:
        def sinh(self, idx: torch.Tensor, max_new_tokens: int, **_: object) -> torch.Tensor:
            moi = torch.randint(3, 20, (1, max_new_tokens), device=idx.device)
            return torch.cat([idx, moi], dim=1)

    prompt = [PromptDiagnostic(id="p", nhom="x", text="mở đầu")]
    profile = [ProfileSinh("a", 1.0, None, None, 1.0)]
    a = chay_benchmark(ModelGia(), TokGia(), prompt, profile, [123], 6, "cpu")
    b = chay_benchmark(ModelGia(), TokGia(), prompt, profile, [123], 6, "cpu")
    assert a[0].output == b[0].output


def test_chay_benchmark_seed_khac_co_the_cho_mau_khac() -> None:
    import torch

    class TokGia:
        def encode(self, text: str) -> list[int]:
            return [7]

        def decode(self, ids: list[int]) -> str:
            return " ".join(str(x) for x in ids)

    class ModelGia:
        def sinh(self, idx: torch.Tensor, max_new_tokens: int, **_: object) -> torch.Tensor:
            moi = torch.randint(3, 100, (1, max_new_tokens), device=idx.device)
            return torch.cat([idx, moi], dim=1)

    prompt = [PromptDiagnostic(id="p", nhom="x", text="mở đầu")]
    profile = [ProfileSinh("a", 1.0, None, None, 1.0)]
    mau = chay_benchmark(ModelGia(), TokGia(), prompt, profile, [1, 2], 8, "cpu")
    assert mau[0].output != mau[1].output
