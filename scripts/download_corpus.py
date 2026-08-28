"""Tải corpus tiếng Việt về data/raw/*.jsonl.

Dùng:
    python scripts/download_corpus.py --source wikipedia --target-gb 1.5
    python scripts/download_corpus.py --source culturax  --target-gb 6

Ghi chú về nguồn:
* wikipedia  — `wikimedia/wikipedia`, config `20231101.vi`. KHÔNG cần đăng nhập.
               Sạch nhất, ~1,5GB. Đủ dư để train tokenizer ở Chặng 1.
* culturax   — `uonlp/CulturaX`, subset `vi`. Đã lọc sẵn, đây là nguồn chính cho
               2,2 tỷ token. CÓ GATE: phải đăng nhập HF và bấm đồng ý điều khoản
               trên trang dataset trước, rồi `hf auth login` (bản cũ: `huggingface-cli login`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.data import normalize_text  # noqa: E402

SOURCES = {
    "wikipedia": {"path": "wikimedia/wikipedia", "name": "20231101.vi"},
    "culturax": {"path": "uonlp/CulturaX", "name": "vi"},
}

# Mỗi shard ~256MB văn bản: đủ nhỏ để mở lại bằng tay, đủ lớn để không sinh nghìn file.
SHARD_BYTES = 256 * 1024 * 1024


def dem_da_co(out_dir: Path, source: str) -> tuple[int, int, int]:
    """Đếm những gì lần tải trước để lại: (số document, số byte văn bản, shard kế tiếp).

    Ctrl+C giữa chừng là chuyện bình thường khi tải vài GB, nên lần chạy sau phải biết
    bỏ qua đúng phần đã có. Stream của CulturaX có thứ tự tất định và bộ lọc `min_chars`
    cũng tất định, nên "bỏ qua N document đầu đã nhận" tái lập chính xác điểm dừng cũ.
    """
    n_docs = n_bytes = 0
    chi_so: list[int] = []
    for path in sorted(out_dir.glob(f"{source}_vi_*.jsonl")):
        try:
            chi_so.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text")
                if isinstance(text, str) and text:
                    n_docs += 1
                    n_bytes += len(text.encode("utf-8"))
    return n_docs, n_bytes, (max(chi_so) + 1 if chi_so else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), default="wikipedia")
    parser.add_argument("--target-gb", type=float, default=1.5)
    parser.add_argument("--out-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--min-chars", type=int, default=200, help="bỏ document quá ngắn")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="chạy tiếp lần tải bị ngắt: bỏ qua số document đã có trong --out-dir "
        "thay vì ghi trùng lại từ đầu.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="dừng sau N document. Dùng để thử 30 giây xem mạng và quyền truy cập có ổn "
        "trước khi cam kết tải vài GB.",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Thiếu gói: pip install datasets", file=sys.stderr)
        return 1

    spec = SOURCES[args.source]
    target_bytes = int(args.target_gb * 1024**3)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # streaming=True: không tải nguyên bộ 68GB về đĩa rồi mới lọc.
    try:
        stream = load_dataset(spec["path"], spec["name"], split="train", streaming=True)
    except Exception as exc:  # noqa: BLE001 - lỗi mạng/quyền/tên config đều rơi vào đây
        print(f"Không mở được {spec['path']} ({spec['name']}): {exc}\n", file=sys.stderr)
        if args.source == "culturax":
            print(
                "CulturaX có gate. Ba bước, làm một lần:\n"
                "  1. Mở https://huggingface.co/datasets/uonlp/CulturaX và bấm đồng ý điều khoản\n"
                "  2. pip install -U huggingface_hub\n"
                "  3. hf auth login   (dán token đọc từ https://huggingface.co/settings/tokens)",
                file=sys.stderr,
            )
        else:
            try:
                from datasets import get_dataset_config_names

                ten = [n for n in get_dataset_config_names(spec["path"]) if n.endswith(".vi")]
                print(f"Config tiếng Việt hiện có: {ten}", file=sys.stderr)
            except Exception:  # noqa: BLE001
                pass
        return 1

    if args.resume:
        da_co_docs, da_co_bytes, shard_dau = dem_da_co(args.out_dir, args.source)
        print(f"Chạy tiếp: đã có {da_co_docs:,} doc / {da_co_bytes / 1024**3:.2f} GB")
        if da_co_bytes >= target_bytes:
            print("Đã đủ mục tiêu, không cần tải thêm.")
            return 0
    else:
        da_co_docs, da_co_bytes, shard_dau = 0, 0, 0

    total = da_co_bytes
    n_docs = da_co_docs
    bo_qua = 0
    shard_idx = shard_dau
    shard_bytes = 0
    fh = None
    try:
        for row in stream:
            text = normalize_text(row.get("text", ""))
            if len(text) < args.min_chars:
                continue
            if bo_qua < da_co_docs:
                bo_qua += 1
                if bo_qua % 100_000 == 0:
                    print(f"   bỏ qua {bo_qua:,}/{da_co_docs:,} doc đã có", flush=True)
                continue
            if fh is None:
                path = args.out_dir / f"{args.source}_vi_{shard_idx:04d}.jsonl"
                fh = path.open("w", encoding="utf-8")
                print(f"-> {path.name}")
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            size = len(text.encode("utf-8"))
            total += size
            shard_bytes += size
            n_docs += 1
            if n_docs % 10_000 == 0:
                print(f"   {n_docs:,} doc | {total / 1024**3:.2f} GB", flush=True)
            if shard_bytes >= SHARD_BYTES:
                fh.close()
                fh = None
                shard_bytes = 0
                shard_idx += 1
            if total >= target_bytes:
                break
            if args.max_docs is not None and n_docs >= args.max_docs:
                print(f"Dừng sớm theo --max-docs={args.max_docs}.")
                break
    finally:
        if fh is not None:
            fh.close()

    print(f"Xong: {n_docs:,} document, {total / 1024**3:.2f} GB -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
