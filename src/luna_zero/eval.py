"""Final eval trên dữ liệu held-out chưa từng dùng để train hay chọn checkpoint."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from luna_zero import config
from luna_zero.data import doc_hash, normalize_text


@dataclass(frozen=True)
class KiemTraTachBiet:
    n_eval_docs: int
    n_eval_unique: int
    n_train_docs_quet: int
    n_trung_train: int
    n_trung_noi_bo_eval: int
    vi_du_trung: tuple[str, ...]

    @property
    def hop_le(self) -> bool:
        return self.n_trung_train == 0 and self.n_trung_noi_bo_eval == 0 and self.n_eval_docs > 0


def kiem_tra_tach_biet(
    eval_docs: Iterable[str],
    train_docs: Iterable[str],
    max_examples: int = config.EVAL.max_overlap_examples,
) -> KiemTraTachBiet:
    """Chứng minh held-out không trùng nguyên document với corpus đã dùng trước đó.

    `train_docs` phải là TOÀN BỘ raw corpus đã dùng để tạo cả train.bin lẫn val.bin.
    Quét cả hai phía là cố ý: val đã tham gia chọn `best.pt`, nên cũng không được lọt
    vào final eval. Chỉ giữ hash của tập eval trong RAM; corpus hàng GB được quét dòng.
    """
    eval_hashes: dict[str, str] = {}
    n_eval = 0
    n_dup = 0
    for text in eval_docs:
        clean = normalize_text(text)
        if not clean:
            continue
        n_eval += 1
        h = doc_hash(clean)
        if h in eval_hashes:
            n_dup += 1
        else:
            eval_hashes[h] = clean[:160]

    overlap: set[str] = set()
    examples: list[str] = []
    n_train = 0
    for text in train_docs:
        clean = normalize_text(text)
        if not clean:
            continue
        n_train += 1
        h = doc_hash(clean)
        if h in eval_hashes and h not in overlap:
            overlap.add(h)
            if len(examples) < max_examples:
                examples.append(eval_hashes[h])

    return KiemTraTachBiet(
        n_eval_docs=n_eval,
        n_eval_unique=len(eval_hashes),
        n_train_docs_quet=n_train,
        n_trung_train=len(overlap),
        n_trung_noi_bo_eval=n_dup,
        vi_du_trung=tuple(examples),
    )


def bat_buoc_tap_eval_sach(kiem: KiemTraTachBiet) -> None:
    """Final eval phải từ chối chạy nếu chưa chứng minh được tách biệt."""
    if kiem.n_eval_docs == 0:
        raise ValueError("Tập eval rỗng, không có gì để đo.")
    if kiem.n_trung_noi_bo_eval:
        raise ValueError(
            f"Tập eval có {kiem.n_trung_noi_bo_eval} document trùng nội bộ; "
            "lặp mẫu sẽ làm metric bị cân sai."
        )
    if kiem.n_trung_train:
        vi_du = f" Ví dụ: {kiem.vi_du_trung[0]!r}" if kiem.vi_du_trung else ""
        raise ValueError(
            f"CONTAMINATION: {kiem.n_trung_train} document eval đã có trong corpus "
            f"train/validation.{vi_du}"
        )


def cua_so_du_doan(ids: list[int], block_size: int) -> Iterator[tuple[list[int], list[int]]]:
    """Cắt (x, y) sao cho y luôn là token kế tiếp, kể cả ở ranh giới cửa sổ."""
    if block_size < 1:
        raise ValueError("block_size phải >= 1")
    for start in range(0, max(0, len(ids) - 1), block_size):
        chunk = ids[start : start + block_size + 1]
        if len(chunk) >= 2:
            yield chunk[:-1], chunk[1:]


@dataclass(frozen=True)
class KetQuaLoss:
    nll: float
    perplexity: float
    n_tokens: int
    n_windows: int
    n_docs: int


def do_loss_heldout(
    model: Any,
    tokenizer: Any,
    texts: Iterable[str],
    device: str,
    batch_size: int = config.EVAL.batch_size,
    max_docs: int | None = None,
    max_windows: int | None = None,
) -> KetQuaLoss:
    """Đo NLL có trọng số theo token trên held-out raw documents.

    Không dùng `model(..., targets).loss` theo trung bình từng batch rồi lấy trung bình
    lần nữa, vì batch cuối/ngắn hơn sẽ bị cân ngang batch đầy đủ. Ở đây cộng tổng CE
    trên mọi target hợp lệ rồi chia đúng tổng số token được đo.
    """
    if batch_size < 1:
        raise ValueError("batch_size phải >= 1")

    import torch
    from torch.nn import functional as F

    model.eval()
    windows: list[tuple[list[int], list[int]]] = []
    tong_loss = 0.0
    n_tokens = 0
    n_windows = 0
    n_docs = 0

    def flush() -> None:
        nonlocal tong_loss, n_tokens, n_windows
        if not windows:
            return
        dai = max(len(x) for x, _ in windows)
        x = torch.full((len(windows), dai), config.PAD_ID, dtype=torch.long, device=device)
        y = torch.full((len(windows), dai), -1, dtype=torch.long, device=device)
        for i, (xi, yi) in enumerate(windows):
            x[i, : len(xi)] = torch.tensor(xi, dtype=torch.long, device=device)
            y[i, : len(yi)] = torch.tensor(yi, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(x).logits
            tong_loss += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-1,
                    reduction="sum",
                ).item()
            )
        n_tokens += int((y != -1).sum().item())
        n_windows += len(windows)
        windows.clear()

    dung = False
    for text in texts:
        clean = normalize_text(text)
        if not clean:
            continue
        n_docs += 1
        ids = tokenizer.encode_document(clean)
        for pair in cua_so_du_doan(ids, model.cfg.block_size):
            windows.append(pair)
            if len(windows) >= batch_size:
                flush()
            if max_windows is not None and n_windows + len(windows) >= max_windows:
                dung = True
                break
        if dung or (max_docs is not None and n_docs >= max_docs):
            break
    flush()

    if n_tokens == 0:
        raise ValueError("Tập eval không tạo được target token nào.")
    nll = tong_loss / n_tokens
    return KetQuaLoss(
        nll=nll,
        perplexity=math.exp(nll),
        n_tokens=n_tokens,
        n_windows=n_windows,
        n_docs=n_docs,
    )


def file_fingerprint(path: Path) -> str:
    """Vân tay byte-level của file nguồn/output để report eval truy nguyên được."""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_jsonl_eval_strict(path: Path, field: str = "text") -> Iterator[str]:
    """Đọc source eval theo chế độ strict; dòng hỏng không được phép biến mất âm thầm."""
    import json

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON hỏng ở {path}:{line_no}: {exc.msg}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Dòng {line_no} của {path} không phải JSON object.")
            value = obj.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Dòng {line_no} của {path} thiếu trường chuỗi không rỗng {field!r}."
                )
            yield value


@dataclass(frozen=True)
class KetQuaTaoHeldout:
    source_path: str
    source_fingerprint: str
    output_path: str
    output_fingerprint: str
    n_docs: int
    n_bytes: int
    kiem_tra_tach_biet: KiemTraTachBiet


def tao_corpus_heldout(
    source: Path,
    output: Path,
    train_corpus_dir: Path,
    field: str = "text",
) -> KetQuaTaoHeldout:
    """Tạo final held-out từ một JSONL bên ngoài corpus đã dùng để train/chọn model.

    Hàm từ chối nguồn nằm trong `train_corpus_dir`, quét exact-overlap sau NFC trên
    TOÀN raw corpus cũ, rồi mới ghi output atomically. Đây là bằng chứng tách biệt ở
    mức nguyên document; near-duplicate vẫn phải được mô tả là giới hạn của phép đo.
    """
    import os

    from luna_zero.data import iter_corpus, write_jsonl

    source = source.resolve()
    output = output.resolve()
    train_corpus_dir = train_corpus_dir.resolve()

    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy source eval: {source}")
    if source == output:
        raise ValueError("Source eval và output không được là cùng một file.")
    if source.is_relative_to(train_corpus_dir):
        raise ValueError(
            "Source eval nằm bên trong raw corpus đã dùng để train/validation; "
            "hãy dùng một nguồn độc lập bên ngoài thư mục đó."
        )
    if not train_corpus_dir.is_dir():
        raise FileNotFoundError(
            "Không có raw corpus cũ để chứng minh held-out tách biệt: " f"{train_corpus_dir}"
        )

    source_fingerprint = file_fingerprint(source)
    kiem = kiem_tra_tach_biet(
        iter_jsonl_eval_strict(source, field),
        iter_corpus(train_corpus_dir),
    )
    bat_buoc_tap_eval_sach(kiem)

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    try:
        n_docs, n_bytes = write_jsonl(tmp, iter_jsonl_eval_strict(source, field))
        if n_docs != kiem.n_eval_docs:
            raise RuntimeError(
                "Số document thay đổi giữa lúc kiểm contamination và lúc ghi output; "
                "source có thể đã bị sửa trong khi chạy."
            )
        if file_fingerprint(source) != source_fingerprint:
            raise RuntimeError(
                "Source eval đã thay đổi trong lúc kiểm contamination; từ chối ghi "
                "một output chưa được chứng minh tách biệt."
            )
        os.replace(tmp, output)
    finally:
        tmp.unlink(missing_ok=True)

    return KetQuaTaoHeldout(
        source_path=str(source),
        source_fingerprint=source_fingerprint,
        output_path=str(output),
        output_fingerprint=file_fingerprint(output),
        n_docs=n_docs,
        n_bytes=n_bytes,
        kiem_tra_tach_biet=kiem,
    )


@dataclass(frozen=True)
class ProvenanceHuman:
    """Provenance tối thiểu để final loss có thể kiểm tra lại nguồn dữ liệu."""

    source_name: str
    source_url: str
    retrieved_at: str
    license_or_permission: str
    notes: str = ""


def tao_corpus_human_heldout(
    source: Path,
    output: Path,
    train_corpus_dir: Path,
    provenance: ProvenanceHuman,
    human_authored_confirmed: bool,
    field: str = "text",
) -> dict[str, Any]:
    """Khóa corpus human-authored trước khi nhìn final NLL/PPL.

    Việc văn bản do người viết là một xác nhận provenance của người chuẩn bị dữ liệu,
    không thể suy ra từ nội dung. Code chỉ chấp nhận xác nhận tường minh và lưu nó cùng
    fingerprint; exact-overlap với raw corpus vẫn được chứng minh bằng phép quét riêng.
    """
    if not human_authored_confirmed:
        raise ValueError("Phải xác nhận nguồn là human-authored trước khi khóa final eval.")
    for ten, value in (
        ("source_name", provenance.source_name),
        ("source_url", provenance.source_url),
        ("retrieved_at", provenance.retrieved_at),
        ("license_or_permission", provenance.license_or_permission),
    ):
        if not value.strip():
            raise ValueError(f"Provenance thiếu {ten}.")

    ket_qua = tao_corpus_heldout(
        source=source,
        output=output,
        train_corpus_dir=train_corpus_dir,
        field=field,
    )
    return {
        "schema_version": 1,
        "purpose": "final_human_heldout",
        "human_authored_confirmed": True,
        "used_for_training": False,
        "used_for_checkpoint_or_sampling_selection": False,
        "eligible_for_final_loss_claims": True,
        "provenance": asdict(provenance),
        "source_path": ket_qua.source_path,
        "source_fingerprint": ket_qua.source_fingerprint,
        "output_path": ket_qua.output_path,
        "output_fingerprint": ket_qua.output_fingerprint,
        "n_docs": ket_qua.n_docs,
        "n_bytes": ket_qua.n_bytes,
        "kiem_tra_tach_biet": asdict(ket_qua.kiem_tra_tach_biet),
        "limitations": (
            "Exact document non-overlap sau NFC đã được kiểm; chưa chứng minh sạch "
            "near-duplicate/paraphrase. Human-authored là xác nhận provenance, không "
            "phải phân loại tự động từ nội dung."
        ),
    }


def doc_meta_human_final(meta_path: Path, eval_path: Path) -> dict[str, Any]:
    """Đọc metadata final-human và chặn đổi corpus sau khi đã khóa."""
    import json

    raw = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    if raw.get("purpose") != "final_human_heldout":
        raise ValueError("Metadata không phải final_human_heldout.")
    if raw.get("human_authored_confirmed") is not True:
        raise ValueError("Metadata chưa xác nhận human-authored.")
    if raw.get("used_for_checkpoint_or_sampling_selection") is not False:
        raise ValueError("Final human corpus không được tham gia chọn checkpoint/sampling.")
    if raw.get("eligible_for_final_loss_claims") is not True:
        raise ValueError("Metadata không cho phép dùng làm final loss.")
    actual = file_fingerprint(Path(eval_path))
    if raw.get("output_fingerprint") != actual:
        raise ValueError("final_human.jsonl đã đổi sau khi metadata được khóa.")
    return raw
