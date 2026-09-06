"""Đánh giá sinh văn bản có provenance, seed tái lập và khóa final benchmark."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from luna_zero import config
from luna_zero.data import iter_corpus, normalize_text
from luna_zero.do_sinh import DoLap, do_lap


@dataclass(frozen=True)
class ProfileSinh:
    ten: str
    temperature: float
    top_k: int | None
    top_p: float | None
    phat_lap: float


@dataclass(frozen=True)
class PromptDiagnostic:
    id: str
    nhom: str
    text: str


@dataclass(frozen=True)
class MauSinh:
    prompt_id: str
    prompt: str
    nhom: str
    seed: int
    profile: str
    output: str
    eos: bool
    n_token_sinh: int
    lap: DoLap

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["lap"] = asdict(self.lap)
        return d


def fingerprint_file(path: Path) -> str:
    """BLAKE2b-128 đọc theo luồng để định danh tokenizer hoặc checkpoint."""
    h = hashlib.blake2b(digest_size=16)
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_json(data: dict[str, Any]) -> str:
    """Fingerprint semantic, không phụ thuộc khoảng trắng hay xuống dòng JSON."""
    payload = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _doc_prompts(raw: dict[str, Any]) -> list[PromptDiagnostic]:
    prompts_raw = raw.get("prompts")
    if not isinstance(prompts_raw, list) or not prompts_raw:
        raise ValueError("Prompt suite rỗng.")

    prompts: list[PromptDiagnostic] = []
    ids: set[str] = set()
    texts: set[str] = set()
    for item in prompts_raw:
        if not isinstance(item, dict):
            raise ValueError("Mỗi prompt phải là object JSON.")
        prompt = PromptDiagnostic(
            id=str(item.get("id", "")).strip(),
            nhom=str(item.get("nhom", "")).strip(),
            text=str(item.get("text", "")).strip(),
        )
        if not prompt.id or not prompt.nhom or not prompt.text:
            raise ValueError("Prompt phải có id, nhom và text không rỗng.")
        if prompt.id in ids:
            raise ValueError(f"Trùng prompt id: {prompt.id}")
        if prompt.text in texts:
            raise ValueError(f"Trùng prompt text: {prompt.text}")
        ids.add(prompt.id)
        texts.add(prompt.text)
        prompts.append(prompt)
    return prompts


def doc_suite(path: Path) -> tuple[dict[str, Any], list[PromptDiagnostic]]:
    """Đọc diagnostic suite; suite này được phép dùng để tune nhiều lần."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("purpose") != "diagnostic":
        raise ValueError("Prompt suite phải khai purpose='diagnostic'.")
    if raw.get("eligible_for_final_claims") is not False:
        raise ValueError("Diagnostic suite phải khai eligible_for_final_claims=false.")
    if raw.get("used_for_sampling_tuning") is not True:
        raise ValueError("Diagnostic suite phải khai used_for_sampling_tuning=true.")
    return raw, _doc_prompts(raw)


def profile_mac_dinh() -> ProfileSinh:
    """Profile đã được chọn trên diagnostic suite; luật sampling vẫn chỉ ở model."""
    return ProfileSinh(
        ten="mac_dinh",
        temperature=config.SINH.temperature,
        top_k=config.SINH.top_k,
        top_p=config.SINH.top_p,
        phat_lap=config.SINH.phat_lap,
    )


def profiles_mac_dinh() -> list[ProfileSinh]:
    """Hai profile diagnostic: cấu hình hiện tại và cấu hình sample lịch sử."""
    return [
        profile_mac_dinh(),
        ProfileSinh(
            ten="sample_lich_su",
            temperature=config.SINH_EVAL.legacy_temperature,
            top_k=config.SINH_EVAL.legacy_top_k,
            top_p=config.SINH_EVAL.legacy_top_p,
            phat_lap=config.SINH_EVAL.legacy_phat_lap,
        ),
    ]


def final_run_config() -> dict[str, Any]:
    """Spec final lấy từ config duy nhất; suite chỉ khóa fingerprint của spec này."""
    return {
        "checkpoint_step": config.SINH_EVAL.final_checkpoint_step,
        "checkpoint_fingerprint": config.SINH_EVAL.final_checkpoint_fingerprint,
        "seeds": list(config.SINH_EVAL.seeds),
        "profile": asdict(profile_mac_dinh()),
        "so_token": config.SINH.so_token,
    }


def final_run_fingerprint() -> str:
    return fingerprint_json(final_run_config())


def doc_suite_final(
    path: Path,
    expected_suite_fingerprint: str | None = None,
) -> tuple[dict[str, Any], list[PromptDiagnostic]]:
    """Đọc final suite và bắt mọi thay đổi suite/profile/seed ngoài ý muốn."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("purpose") != "final":
        raise ValueError("Final suite phải khai purpose='final'.")
    if raw.get("eligible_for_final_claims") is not True:
        raise ValueError("Final suite phải khai eligible_for_final_claims=true.")
    if raw.get("used_for_sampling_tuning") is not False:
        raise ValueError("Final suite phải khai used_for_sampling_tuning=false.")
    if raw.get("run_config_fingerprint") != final_run_fingerprint():
        raise ValueError("Cấu hình generation final đã đổi sau khi suite được khóa.")

    expected = (
        config.SINH_EVAL.final_suite_fingerprint
        if expected_suite_fingerprint is None
        else expected_suite_fingerprint
    )
    if fingerprint_json(raw) != expected:
        raise ValueError("Final prompt suite đã đổi sau khi fingerprint được khóa.")
    return raw, _doc_prompts(raw)


def bat_overlap_prompt(
    final_prompts: Iterable[PromptDiagnostic],
    diagnostic_prompts: Iterable[PromptDiagnostic],
) -> None:
    """Final prompts không được tái dùng prompt đã tham gia tuning."""
    diag = {normalize_text(p.text) for p in diagnostic_prompts}
    trung = [p.id for p in final_prompts if normalize_text(p.text) in diag]
    if trung:
        raise ValueError(f"Final prompt trùng diagnostic: {', '.join(trung)}")


def quet_prompt_trong_corpus(
    prompts: Iterable[PromptDiagnostic],
    corpus_dir: Path,
) -> tuple[int, dict[str, int]]:
    """Đếm exact prompt substring trong raw corpus sau NFC; final yêu cầu toàn 0."""
    prompt_list = list(prompts)
    if not prompt_list:
        raise ValueError("Không có prompt để kiểm contamination.")
    texts = [(p.id, normalize_text(p.text)) for p in prompt_list]
    hits = {p.id: 0 for p in prompt_list}
    n_docs = 0
    for doc in iter_corpus(Path(corpus_dir)):
        n_docs += 1
        normalized = normalize_text(doc)
        for prompt_id, text in texts:
            if text in normalized:
                hits[prompt_id] += 1
    return n_docs, hits


def bao_dam_prompt_final_sach(prompts: list[PromptDiagnostic], corpus_dir: Path) -> int:
    """Từ chối final nếu prompt đã xuất hiện nguyên văn trong corpus train."""
    n_docs, hits = quet_prompt_trong_corpus(prompts, corpus_dir)
    if n_docs == 0:
        raise ValueError("Không quét được document nào trong raw corpus train.")
    contaminated = {k: v for k, v in hits.items() if v}
    if contaminated:
        mo_ta = ", ".join(f"{k}={v}" for k, v in sorted(contaminated.items()))
        raise ValueError(f"Final prompt xuất hiện trong corpus train: {mo_ta}")
    return n_docs


def kiem_checkpoint_final(step: int, fingerprint: str) -> None:
    """Final benchmark chỉ được chạy trên đúng model object đã chọn ở diagnostic."""
    if step != config.SINH_EVAL.final_checkpoint_step:
        raise ValueError(
            f"Checkpoint final phải ở bước {config.SINH_EVAL.final_checkpoint_step:,}, "
            f"đang là {step:,}."
        )
    if fingerprint != config.SINH_EVAL.final_checkpoint_fingerprint:
        raise ValueError("Checkpoint final đúng step nhưng sai fingerprint.")


def duong_dan_bao_cao_final(
    out_dir: Path,
    step: int,
    checkpoint_fingerprint: str,
    suite_fingerprint: str,
) -> Path:
    """Đường dẫn deterministic để cùng final spec không bị chạy âm thầm lần hai."""
    key = f"{checkpoint_fingerprint}:{suite_fingerprint}".encode("ascii")
    run_id = hashlib.blake2b(key, digest_size=6).hexdigest()
    return Path(out_dir) / f"generation_final_step_{step:07d}_{run_id}.json"


def bao_dam_chua_chay_final(path: Path) -> None:
    if Path(path).exists():
        raise FileExistsError(
            f"Final generation đã có artifact {path}; không chạy lại cùng spec/checkpoint."
        )


def tom_tat(mau: list[MauSinh]) -> dict[str, dict[str, float | int]]:
    """Tổng hợp metric lặp thô theo profile; không gán nhãn chất lượng."""
    ket_qua: dict[str, dict[str, float | int]] = {}
    for profile in sorted({m.profile for m in mau}):
        nhom = [m for m in mau if m.profile == profile]
        ket_qua[profile] = {
            "n_samples": len(nhom),
            "eos_rate": sum(m.eos for m in nhom) / len(nhom),
            "distinct_1_mean": statistics.fmean(m.lap.distinct_1 for m in nhom),
            "distinct_2_mean": statistics.fmean(m.lap.distinct_2 for m in nhom),
            "distinct_4_mean": statistics.fmean(m.lap.distinct_4 for m in nhom),
            "max_4gram_repeat": max(m.lap.lap_dai_nhat for m in nhom),
        }
    return ket_qua


def chay_benchmark(
    model: Any,
    tok: Any,
    prompts: list[PromptDiagnostic],
    profiles: list[ProfileSinh],
    seeds: list[int],
    so_token: int,
    device: str,
) -> list[MauSinh]:
    """Sinh ma trận prompt × profile × seed bằng luật duy nhất trong model.sinh()."""
    import torch

    if so_token < 1:
        raise ValueError("so_token phải >= 1")
    if not seeds:
        raise ValueError("Cần ít nhất một seed")

    mau: list[MauSinh] = []
    for prompt in prompts:
        ids = tok.encode(prompt.text)
        x = torch.tensor([[config.BOS_ID, *ids]], dtype=torch.long, device=device)
        for profile in profiles:
            for seed in seeds:
                torch.manual_seed(seed)
                ra = model.sinh(
                    x,
                    max_new_tokens=so_token,
                    temperature=profile.temperature,
                    top_k=profile.top_k,
                    top_p=profile.top_p,
                    phat_lap=profile.phat_lap,
                    dung_o_eos=True,
                )
                ids_moi = ra[0, x.size(1) :].tolist()
                eos = config.EOS_ID in ids_moi
                if eos:
                    ids_moi = ids_moi[: ids_moi.index(config.EOS_ID)]
                output = tok.decode(ids_moi)
                mau.append(
                    MauSinh(
                        prompt_id=prompt.id,
                        prompt=prompt.text,
                        nhom=prompt.nhom,
                        seed=seed,
                        profile=profile.ten,
                        output=output,
                        eos=eos,
                        n_token_sinh=len(ids_moi),
                        lap=do_lap(output),
                    )
                )
    return mau
