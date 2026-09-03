"""Giao diện chat cục bộ của Luna Zero — mở trình duyệt, nói chuyện với checkpoint.

python scripts/giao_dien.py                 # nạp checkpoint mới nhất
python scripts/giao_dien.py --gia-lap       # chưa có checkpoint: xem thử giao diện
python scripts/giao_dien.py --best --device cpu --port 8080

Chạy được SONG SONG với vòng train (nó chỉ đọc file checkpoint), nhưng nạp model lên
GPU đang train sẽ ăn thêm ~0,5GB VRAM — dùng `--device cpu` nếu VRAM đang sát trần.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luna_zero import config  # noqa: E402
from luna_zero.giao_dien import NguonGiaLap, nguon_tu_checkpoint, tao_server  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    p.add_argument("--tokenizer", type=Path, default=config.TOKENIZER_PATH)
    p.add_argument("--device", default=None)
    p.add_argument("--best", action="store_true", help="dùng best.pt thay vì bản mới nhất")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="đổi sang 0.0.0.0 là mở cho cả mạng LAN — giao diện KHÔNG có xác thực",
    )
    p.add_argument(
        "--gia-lap",
        action="store_true",
        help="phát chữ có sẵn thay vì nạp model; xem được giao diện khi chưa có checkpoint",
    )
    p.add_argument("--khong-mo-trinh-duyet", action="store_true")
    args = p.parse_args()

    if args.gia_lap:
        nguon = NguonGiaLap()
    else:
        try:
            nguon = nguon_tu_checkpoint(args.checkpoint_dir, args.tokenizer, args.device, args.best)
        except FileNotFoundError as e:
            # Trước Chặng 3 thì đây là trạng thái BÌNH THƯỜNG, không phải sự cố —
            # nói rõ lối đi tiếp thay vì để người dùng đoán.
            print(f"{e}\nChưa train xong thì xem giao diện bằng: --gia-lap", file=sys.stderr)
            return 1

    server = tao_server(nguon, host=args.host, port=args.port)
    dia_chi = f"http://{args.host}:{args.port}"
    print(f"Luna Zero đang nghe ở {dia_chi}  (Ctrl+C để dừng)")
    for khoa, gt in nguon.trang_thai().items():
        print(f"  {khoa:10} {gt}")

    if not args.khong_mo_trinh_duyet:
        threading.Timer(0.5, webbrowser.open, args=[dia_chi]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDừng.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
