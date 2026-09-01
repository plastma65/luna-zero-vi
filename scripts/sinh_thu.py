"""Sinh văn bản từ checkpoint — xem model đang viết ra cái gì.

python scripts/sinh_thu.py --checkpoint-dir C:\\Luna_checkpoints
python scripts/sinh_thu.py --moi "Hà Nội là" --so-mau 3 --temperature 0.8
python scripts/sinh_thu.py --device cpu     # chạy được không cần GPU
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.checkpoint import CheckpointManager  # noqa: E402
from luna_zero.tokenizer import LunaTokenizer  # noqa: E402

MOI_MAC_DINH = [
    "Hà Nội là",
    "Theo báo cáo mới nhất,",
    "Cách nấu phở bò ngon là",
    "Trong lĩnh vực công nghệ thông tin,",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    p.add_argument("--tokenizer", type=Path, default=config.TOKENIZER_PATH)
    p.add_argument("--moi", action="append", default=None, help="câu mở đầu, lặp được")
    p.add_argument("--so-token", type=int, default=120)
    p.add_argument("--so-mau", type=int, default=1, help="số bản sinh cho mỗi câu mở đầu")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=0.92, help="nucleus; 0 để tắt")
    p.add_argument(
        "--phat-lap",
        type=float,
        default=1.15,
        help="1.0 = tắt. Cắt vòng lặp kiểu 'bảo đảm, bảo đảm, bảo đảm'.",
    )
    p.add_argument(
        "--luu",
        action="store_true",
        help="lưu kết quả vào artifacts/samples/step_XXXXXXX.txt để so mốc về sau",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--best", action="store_true", help="dùng best.pt thay vì bản mới nhất")
    args = p.parse_args()

    import torch

    from luna_zero.config import chon_thiet_bi
    from luna_zero.model import LunaZeroGPT

    device = chon_thiet_bi(args.device)
    tok = LunaTokenizer.load(args.tokenizer)

    manager = CheckpointManager(args.checkpoint_dir)
    duong_dan = args.checkpoint_dir / "best.pt" if args.best else manager.latest_path()
    if duong_dan is None or not duong_dan.exists():
        print(f"Không tìm thấy checkpoint trong {args.checkpoint_dir}", file=sys.stderr)
        return 1
    blob = torch.load(duong_dan, map_location="cpu", weights_only=False)

    # Dựng lại model theo ĐÚNG cấu hình đã lưu trong checkpoint, không theo config hiện
    # tại. Nếu ai sửa config.py sau khi train, nạp theo config mới sẽ lệch hình dạng —
    # hoặc tệ hơn, khớp hình dạng nhưng sai ý nghĩa.
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

    buoc = blob["state"]["step"]
    val = blob["state"]["best_val_loss"]
    print(f"Checkpoint : {duong_dan.name} | bước {buoc:,} | val loss tốt nhất {val:.4f}")
    print(
        f"Thiết bị   : {device} | temp {args.temperature} | top-p {args.top_p} "
        f"| top-k {args.top_k} | phạt lặp {args.phat_lap}"
    )

    dong: list[str] = [
        f"# Luna Zero — bước {buoc:,} | val loss {val:.4f}",
        f"# temp {args.temperature} top-p {args.top_p} top-k {args.top_k} "
        f"phạt lặp {args.phat_lap}",
    ]
    for moi in args.moi or MOI_MAC_DINH:
        ids = tok.encode(moi)
        for i in range(args.so_mau):
            x = torch.tensor([[config.BOS_ID, *ids]], dtype=torch.long, device=device)
            ra = model.sinh(
                x,
                max_new_tokens=args.so_token,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p if args.top_p else None,
                phat_lap=args.phat_lap,
            )
            # Bỏ BOS trước khi giải mã, giữ nguyên phần còn lại.
            van_ban = tok.decode(ra[0, 1:].tolist())
            nhan = f'"{moi}"' if args.so_mau == 1 else f'"{moi}" [{i + 1}]'
            khoi = f"\n{'=' * 70}\n{nhan}\n{'-' * 70}\n{van_ban}"
            print(khoi)
            dong.append(khoi)

    if args.luu:
        # Mẫu sinh là file text nhỏ và quý — đúng loại thứ .gitignore cố ý KHÔNG chặn.
        # Giữ lại từng mốc để so tiến bộ, thay vì chỉ nhớ mang máng "hồi đó tệ hơn".
        thu_muc = config.ARTIFACT_DIR / "samples"
        thu_muc.mkdir(parents=True, exist_ok=True)
        dich = thu_muc / f"step_{buoc:07d}.txt"
        dich.write_text("\n".join(dong) + "\n", encoding="utf-8")
        print(f"\nĐã lưu: {dich}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
