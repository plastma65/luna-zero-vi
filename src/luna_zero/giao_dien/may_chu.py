"""Máy chủ web tí hon cho giao diện chat Luna Zero — CHỈ dùng thư viện chuẩn.

Vì sao không FastAPI: nguyên tắc "ít phụ thuộc" của dự án. Cả giao diện này là một
file HTML tự chứa cộng `http.server`, nên chạy được ngay trong .venv hiện có, không
thêm gì vào requirements.txt và không có gì để hỏng khi nâng cấp.

Vì sao stream từng chữ thay vì trả một cục: hiệu ứng ánh trăng chỉ có nghĩa nếu nó
CHẤM DỨT đúng lúc model phát token đầu tiên. Trả một cục thì mây phải quay suốt rồi
văn bản hiện ra nguyên khối — người dùng không phân biệt được "đang nghĩ" với "đã treo".
"""

from __future__ import annotations

import json
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from luna_zero import config
from luna_zero.do_sinh import do_lap, khoang_nguoi_viet

TRANG_HTML = Path(__file__).parent / "trang.html"

# Ký tự thay thế mà bộ giải mã đẻ ra khi chuỗi byte UTF-8 còn dở dang.
KY_TU_DO = "�"


@dataclass(frozen=True)
class ThamSoSinh:
    """Tham số lấy mẫu cho một lượt chat. Mặc định lấy nguyên từ `config.SINH`."""

    so_token: int = config.SINH.so_token
    temperature: float = config.SINH.temperature
    top_p: float = config.SINH.top_p
    top_k: int | None = config.SINH.top_k
    phat_lap: float = config.SINH.phat_lap

    @classmethod
    def tu_json(cls, du_lieu: dict[str, Any]) -> ThamSoSinh:
        """Chỉ nhận đúng những khoá mình biết, và ép kiểu. Trình duyệt gửi gì cũng được
        nên không được tin: một `so_token` dạng chuỗi sẽ làm vòng `range` nổ giữa chừng
        khi đã gửi header đi rồi, lúc đó không còn cách nào báo lỗi tử tế."""
        sach = cls()
        if "so_token" in du_lieu:
            sach = replace(sach, so_token=max(1, min(int(du_lieu["so_token"]), 1000)))
        if "temperature" in du_lieu:
            sach = replace(sach, temperature=max(0.01, float(du_lieu["temperature"])))
        if "top_p" in du_lieu:
            sach = replace(sach, top_p=float(du_lieu["top_p"]))
        if "phat_lap" in du_lieu:
            sach = replace(sach, phat_lap=float(du_lieu["phat_lap"]))
        return sach


class GiaiMaDan:
    """Giải mã token-theo-token mà không đẻ ra ký tự rác.

    Đây là chỗ dễ sai nhất của phần stream, và nó sai ÂM THẦM: BPE mức byte cắt một
    chữ tiếng Việt có dấu thành nhiều byte, nên một token lẻ thường mang NỬA chuỗi
    UTF-8. Gọi `decode` trên từng token riêng sẽ ra "\ufffd" ở giữa câu — trông như model
    viết bậy trong khi nó viết đúng.

    Cách chắc: decode cả tiền tố, phát phần chênh, và GIỮ LẠI đuôi dở dang cho lượt
    sau. Tốn O(n^2) nhưng n cỡ trăm token nên không đáng kể so với một bước forward.
    """

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        self._ids: list[int] = []
        self._da_phat = ""

    def them(self, token_id: int) -> str:
        """Nạp một token, trả về phần văn bản MỚI ổn định (có thể là chuỗi rỗng)."""
        self._ids.append(int(token_id))
        day_du = self._tok.decode(self._ids)
        on_dinh = day_du.rstrip(KY_TU_DO)
        moi = on_dinh[len(self._da_phat) :]
        self._da_phat = on_dinh
        return moi

    @property
    def van_ban(self) -> str:
        return self._da_phat


class NguonSinh(ABC):
    """Nơi văn bản chảy ra. Có hai bản: checkpoint thật và bản giả lập."""

    @abstractmethod
    def trang_thai(self) -> dict[str, Any]:
        """Mô tả ngắn hiện trên thanh tiêu đề: bước train, val loss, thiết bị."""

    @abstractmethod
    def sinh(self, moi: str, tham_so: ThamSoSinh) -> Iterator[str]:
        """Yield từng mẩu văn bản mới. Kết thúc iterator là kết thúc câu trả lời."""


CAU_GIA_LAP = (
    "Đây là bản giả lập của giao diện Luna Zero. Model thật chưa train xong nên chưa có "
    "checkpoint nào để nạp; phần chữ bạn đang đọc do máy chủ phát ra theo nhịp, chỉ để "
    "kiểm tra rằng hiệu ứng ánh trăng tắt đúng lúc token đầu tiên xuất hiện và văn bản "
    "chảy đều tới hết. Chạy lại không kèm cờ giả lập khi Chặng ba xong."
)


class NguonGiaLap(NguonSinh):
    """Phát chữ theo nhịp, không cần torch và không cần checkpoint.

    Giao diện phải kiểm được TRƯỚC khi model có gì để nói, nếu không thì lỗi giao diện
    và lỗi model trộn vào nhau đúng lúc mình mệt nhất — sau bốn ngày train.
    """

    def __init__(self, nhip_giay: float = 0.04, nghi_dau_giay: float = 0.6) -> None:
        self.nhip_giay = nhip_giay
        self.nghi_dau_giay = nghi_dau_giay

    def trang_thai(self) -> dict[str, Any]:
        return {"che_do": "giả lập", "buoc": None, "val_loss": None, "thiet_bi": "—"}

    def sinh(self, moi: str, tham_so: ThamSoSinh) -> Iterator[str]:
        # Nghỉ một nhịp trước token đầu: đó chính là khoảng thời gian mây che trăng.
        time.sleep(self.nghi_dau_giay)
        for tu in CAU_GIA_LAP.split(" "):
            time.sleep(self.nhip_giay)
            yield tu + " "


def nguon_tu_checkpoint(
    checkpoint_dir: Path,
    tokenizer_path: Path,
    device: str | None = None,
    dung_best: bool = False,
) -> NguonSinh:
    """Nạp checkpoint mới nhất và trả về nguồn sinh thật.

    torch chỉ được import ở đây, không ở đầu file: máy chủ và bản giả lập phải chạy
    được trên máy chưa cài torch (và trong CI, nơi không có GPU lẫn checkpoint).
    """
    from dataclasses import replace as _replace

    import torch

    from luna_zero.checkpoint import CheckpointManager
    from luna_zero.config import chon_thiet_bi
    from luna_zero.model import LunaZeroGPT
    from luna_zero.tokenizer import LunaTokenizer

    thiet_bi = chon_thiet_bi(device)
    tok = LunaTokenizer.load(tokenizer_path)

    manager = CheckpointManager(checkpoint_dir)
    duong_dan = checkpoint_dir / "best.pt" if dung_best else manager.latest_path()
    if duong_dan is None or not duong_dan.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint trong {checkpoint_dir}")
    blob = torch.load(duong_dan, map_location="cpu", weights_only=False)

    # Dựng lại theo cấu hình ĐÃ LƯU, không theo config hiện tại — y như sinh_thu.py.
    cfg = _replace(config.MODEL, **blob.get("model_cfg", {}))
    if cfg.vocab_size != tok.vocab_size:
        raise ValueError(
            f"Vocab lệch: checkpoint {cfg.vocab_size:,} vs tokenizer {tok.vocab_size:,}"
        )
    model = LunaZeroGPT(cfg).to(thiet_bi)
    model.load_state_dict(blob["model"])
    model.eval()

    class _NguonCheckpoint(NguonSinh):
        def trang_thai(self) -> dict[str, Any]:
            return {
                "che_do": duong_dan.name,
                "buoc": blob["state"]["step"],
                "val_loss": blob["state"]["best_val_loss"],
                "thiet_bi": thiet_bi,
            }

        def sinh(self, moi: str, tham_so: ThamSoSinh) -> Iterator[str]:
            ids = tok.encode(moi)
            x = torch.tensor([[config.BOS_ID, *ids]], dtype=torch.long, device=thiet_bi)
            giai_ma = GiaiMaDan(tok)
            for tiep in model.sinh_dan(
                x,
                max_new_tokens=tham_so.so_token,
                temperature=tham_so.temperature,
                top_k=tham_so.top_k,
                top_p=tham_so.top_p or None,
                phat_lap=tham_so.phat_lap,
                dung_o_eos=True,
            ):
                token_id = int(tiep[0, 0])
                if token_id == config.EOS_ID:
                    break  # <eos> là ranh giới bài, không phải chữ để in ra
                mau = giai_ma.them(token_id)
                if mau:
                    yield mau

    return _NguonCheckpoint()


def _danh_gia(van_ban: str) -> dict[str, Any]:
    """Chỉ số lặp cho một câu trả lời, so với người viết CÙNG ĐỘ DÀI.

    Cùng phép đo `sinh_thu.py` đang dùng — không viết lại ngưỡng ở đây, vì ngưỡng viết
    lại là đúng cái bẫy đã làm gắn nhãn "LỆCH" nhầm một lần rồi.
    """
    chi_so = do_lap(van_ban)
    p10, _, p90 = khoang_nguoi_viet(chi_so.n_token)
    if chi_so.n_token < 2:
        nhan = "quá ngắn để đo"
    elif chi_so.distinct_2 < p10:
        nhan = f"LẶP (dưới p10={p10:.2f})"
    elif chi_so.distinct_2 > p90:
        nhan = f"đa dạng bất thường (trên p90={p90:.2f})"
    else:
        nhan = "tự nhiên"
    return {**asdict(chi_so), "danh_gia": nhan}


class _Handler(BaseHTTPRequestHandler):
    server_version = "LunaZero"
    # HTTP/1.0 + đóng kết nối: trình duyệt biết câu trả lời hết khi socket đóng, nên
    # không cần Content-Length (không biết trước) cũng không cần khung chunked tự viết.
    protocol_version = "HTTP/1.0"

    nguon: NguonSinh
    khoa: threading.Lock

    def log_message(self, format: str, *args: Any) -> None:
        pass  # im lặng: log mặc định làm nhiễu terminal đang chạy train

    # --- tiện ích ---
    def _tra_json(self, du_lieu: dict[str, Any], ma: int = 200) -> None:
        body = json.dumps(du_lieu, ensure_ascii=False).encode("utf-8")
        self.send_response(ma)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _gui_dong(self, du_lieu: dict[str, Any]) -> None:
        self.wfile.write(json.dumps(du_lieu, ensure_ascii=False).encode("utf-8") + b"\n")
        self.wfile.flush()

    # --- tuyến ---
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = TRANG_HTML.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/trang_thai":
            # Trang lấy giá trị khởi đầu của các thanh trượt TỪ ĐÂY, không viết
            # cứng trong HTML — nếu không thì config.SINH có bản thứ hai bằng JS.
            self._tra_json({**self.nguon.trang_thai(), "mac_dinh": asdict(ThamSoSinh())})
        else:
            self._tra_json({"loi": "không có tuyến này"}, ma=404)

    def do_POST(self) -> None:
        if self.path != "/api/sinh":
            self._tra_json({"loi": "không có tuyến này"}, ma=404)
            return

        n = int(self.headers.get("Content-Length", 0))
        try:
            yeu_cau = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
            moi = str(yeu_cau.get("moi", "")).strip()
            tham_so = ThamSoSinh.tu_json(yeu_cau)
        except (ValueError, TypeError) as e:
            self._tra_json({"loi": f"yêu cầu hỏng: {e}"}, ma=400)
            return
        if not moi:
            self._tra_json({"loi": "chưa có câu mở đầu"}, ma=400)
            return

        # Một lượt sinh một lúc. Hai lượt song song trên cùng model không sai kết quả
        # nhưng chia đôi tốc độ và nhân đôi VRAM — trên máy đang train thì đó là nguy cơ
        # OOM cho chính vòng train, tức là mất công sức thật.
        if not self.khoa.acquire(blocking=False):
            self._tra_json({"loi": "đang trả lời câu trước"}, ma=409)
            return

        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            bat_dau = time.monotonic()
            day_du: list[str] = []
            dau_tien: float | None = None
            for mau in self.nguon.sinh(moi, tham_so):
                if dau_tien is None:
                    dau_tien = time.monotonic() - bat_dau
                day_du.append(mau)
                self._gui_dong({"loai": "chu", "chu": mau})
            van_ban = "".join(day_du)
            self._gui_dong(
                {
                    "loai": "xong",
                    "giay_cho": round(dau_tien or 0.0, 2),
                    "giay_tong": round(time.monotonic() - bat_dau, 2),
                    "chi_so": _danh_gia(van_ban),
                }
            )
        except (BrokenPipeError, ConnectionResetError):
            pass  # người dùng đóng tab giữa chừng, không phải lỗi
        except Exception as e:  # noqa: BLE001 - header đã gửi, chỉ còn cách báo trong dòng
            try:
                self._gui_dong({"loai": "loi", "loi": f"{type(e).__name__}: {e}"})
            except OSError:
                pass
        finally:
            self.khoa.release()


def tao_server(nguon: NguonSinh, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Dựng server đã gắn nguồn sinh. Gọi `.serve_forever()` để chạy.

    Mặc định 127.0.0.1: giao diện này không có xác thực, mở ra 0.0.0.0 là mời cả mạng
    LAN chạy inference trên GPU đang train.
    """
    lop = type("_HandlerDaGan", (_Handler,), {"nguon": nguon, "khoa": threading.Lock()})
    return ThreadingHTTPServer((host, port), lop)
