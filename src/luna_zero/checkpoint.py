"""Lưu và khôi phục trạng thái train — để ngắt giữa chừng rồi chạy tiếp được.

Bốn điều dễ làm sai, mỗi cái đều làm hỏng lần train mà không báo lỗi:

1. **Chỉ lưu trọng số.** Optimizer AdamW mang hai trạng thái động lượng cho từng tham
   số; mất chúng thì khi chạy tiếp, mô hình bị đá ra khỏi quỹ đạo đang hội tụ và loss
   vọt lên. Phải lưu optimizer, bước hiện tại và RNG. Lịch LR của Luna Zero là hàm
   thuần của `step`, nên không có scheduler object riêng để lưu.
2. **Không lưu vị trí dữ liệu.** Chạy tiếp mà đọc lại từ đầu corpus thì mô hình học
   thuộc phần đầu và không bao giờ thấy phần cuối. Đây là lỗi im lặng nhất.
3. **Ghi đè trực tiếp lên file cũ.** Cúp điện đúng lúc đang ghi là mất luôn checkpoint
   duy nhất còn tốt. Ở đây ghi ra `.tmp` rồi `os.replace` — thao tác nguyên tử trên cả
   Windows lẫn Linux, nên file cũ chỉ biến mất khi file mới đã hoàn chỉnh.
4. **Giữ mọi checkpoint.** Bài học Luna cũ: 1,4GB đĩa bốc hơi. Mỗi checkpoint 110M
   tham số nặng ~1,3GB (trọng số + hai trạng thái Adam, fp32), nên có trần cứng.
"""

from __future__ import annotations

import json
import os
import signal
import types
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from luna_zero.config import TRAIN, TrainConfig

MANIFEST_NAME = "manifest.json"
BEST_NAME = "best.pt"
TMP_SUFFIX = ".tmp"


@dataclass(frozen=True)
class TrainState:
    """Mọi thứ cần để chạy tiếp đúng chỗ đã dừng, trừ tensor.

    `data_position` là số CỬA SỔ loader đã tiêu thụ trong hoán vị tất định. Nó phải
    dùng đúng cùng quy ước với `BatchLoader.vi_tri`; không phải số token.
    """

    step: int
    tokens_seen: int
    data_position: int
    best_val_loss: float = float("inf")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainState:
        return cls(**{k: d[k] for k in ("step", "tokens_seen", "data_position", "best_val_loss")})


def _torch_save(obj: Any, path: Path) -> None:
    import torch

    torch.save(obj, path)


def _torch_load(path: Path) -> Any:
    import torch

    # weights_only=False vì payload có cả optimizer/RNG, không chỉ tensor.
    # An toàn ở đây: file do chính máy này ghi ra, không phải tải từ internet.
    return torch.load(path, map_location="cpu", weights_only=False)


def lay_rng_state(loai_thiet_bi: str) -> dict[str, Any]:
    """Chụp RNG cần để resume không đổi quỹ đạo khi code có phép toán ngẫu nhiên."""
    import random

    import numpy as np
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if loai_thiet_bi == "cuda":
        state["torch_device"] = torch.cuda.get_rng_state_all()
    return state


def khoi_phuc_rng_state(state: dict[str, Any], loai_thiet_bi: str) -> None:
    """Khôi phục đúng trạng thái đã chụp bởi :func:`lay_rng_state`."""
    import random

    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if loai_thiet_bi == "cuda" and "torch_device" in state:
        torch.cuda.set_rng_state_all(state["torch_device"])


class CheckpointManager:
    """Quản lý thư mục checkpoint: ghi nguyên tử, xoay vòng, và tìm bản mới nhất.

    `save_fn`/`load_fn` tiêm được để test chạy không cần torch — và quan trọng hơn, để
    test mô phỏng được sự cố ghi giữa chừng.
    """

    def __init__(
        self,
        directory: Path,
        max_keep: int = TRAIN.max_checkpoints_keep,
        save_fn: Callable[[Any, Path], None] = _torch_save,
        load_fn: Callable[[Path], Any] = _torch_load,
    ) -> None:
        if max_keep < 1:
            raise ValueError("max_keep phải >= 1, nếu không thì chẳng còn gì để chạy tiếp")
        self.directory = Path(directory)
        self.max_keep = max_keep
        self._save_fn = save_fn
        self._load_fn = load_fn
        self.directory.mkdir(parents=True, exist_ok=True)

    # --- manifest ----------------------------------------------------------
    @property
    def manifest_path(self) -> Path:
        return self.directory / MANIFEST_NAME

    def _read_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Manifest hỏng không được phép chặn việc chạy tiếp: dựng lại từ file có thật.
            return self._rebuild_manifest()
        return data if isinstance(data, list) else []

    def _rebuild_manifest(self) -> list[dict[str, Any]]:
        entries = []
        for path in sorted(self.directory.glob("ckpt_step_*.pt")):
            try:
                step = int(path.stem.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                continue
            entries.append({"step": step, "file": path.name})
        entries.sort(key=lambda e: e["step"])
        return entries

    def _write_manifest(self, entries: list[dict[str, Any]]) -> None:
        tmp = self.manifest_path.with_suffix(".json" + TMP_SUFFIX)
        tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    # --- lưu ---------------------------------------------------------------
    def save(self, state: TrainState, payload: dict[str, Any], is_best: bool = False) -> Path:
        """Ghi một checkpoint. `payload` chứa state_dict của model/optimizer và RNG."""
        name = f"ckpt_step_{state.step:09d}.pt"
        final = self.directory / name
        tmp = final.with_suffix(".pt" + TMP_SUFFIX)

        blob = {"state": state.to_dict(), **payload}
        try:
            self._save_fn(blob, tmp)
        except BaseException:
            # Dọn file dở dang, nhưng KHÔNG đụng vào checkpoint cũ — nó vẫn là bản tốt.
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, final)

        entries = [e for e in self._read_manifest() if e["step"] != state.step]
        entries.append({"step": state.step, "file": name, "best_val_loss": state.best_val_loss})
        entries.sort(key=lambda e: e["step"])
        self._write_manifest(entries)

        if is_best:
            best_tmp = self.directory / (BEST_NAME + TMP_SUFFIX)
            self._save_fn(blob, best_tmp)
            os.replace(best_tmp, self.directory / BEST_NAME)

        self._rotate()
        return final

    def _rotate(self) -> None:
        """Xoá bớt checkpoint cũ, luôn giữ `max_keep` bản mới nhất. `best.pt` bất khả xâm phạm."""
        entries = self._read_manifest()
        thua = entries[: -self.max_keep] if len(entries) > self.max_keep else []
        for entry in thua:
            (self.directory / entry["file"]).unlink(missing_ok=True)
        if thua:
            self._write_manifest(entries[len(thua) :])

    # --- khôi phục ---------------------------------------------------------
    def latest_path(self) -> Path | None:
        for entry in reversed(self._read_manifest()):
            path = self.directory / entry["file"]
            if path.exists():
                return path
        return None

    def load_latest(self) -> tuple[TrainState, dict[str, Any]] | None:
        """Trả về (trạng thái, payload) của checkpoint mới nhất nạp được, hoặc None.

        Lùi dần về bản cũ hơn nếu bản mới nhất hỏng. Ghi nguyên tử đã ngăn được phần lớn
        trường hợp, nhưng đĩa vẫn có thể hỏng bit lẻ, và `max_keep` bản chỉ có giá trị
        khi ta thật sự chịu dùng tới bản thứ hai. Thà lùi 200 bước còn hơn mất 4 ngày.
        """
        for entry in reversed(self._read_manifest()):
            path = self.directory / entry["file"]
            if not path.exists():
                continue
            try:
                blob = self._load_fn(path)
                state = TrainState.from_dict(blob["state"])
            except Exception as exc:  # noqa: BLE001 - hỏng kiểu gì cũng phải lùi được
                print(f"[checkpoint] {path.name} hỏng ({exc}), lùi về bản trước.", flush=True)
                continue
            payload = {k: v for k, v in blob.items() if k != "state"}
            return state, payload
        return None


class NgatMemMai:
    """Bắt Ctrl+C để dừng SAU khi lưu xong, thay vì giết ngang giữa một bước.

    Nhấn Ctrl+C lần một: đặt cờ, vòng lặp train tự kết thúc bước hiện tại rồi lưu.
    Nhấn lần hai: trả quyền cho hành vi mặc định, thoát ngay (dùng khi thật sự gấp).
    """

    def __init__(self) -> None:
        self.duoc_yeu_cau_dung = False
        self._truoc_do: Any = None

    def __enter__(self) -> NgatMemMai:
        self._truoc_do = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._xu_ly)
        return self

    def _xu_ly(self, signum: int, frame: types.FrameType | None) -> None:
        if self.duoc_yeu_cau_dung:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        self.duoc_yeu_cau_dung = True
        print("\n[Ctrl+C] Sẽ lưu checkpoint rồi dừng. Nhấn lần nữa để thoát ngay.", flush=True)

    def __exit__(self, *exc: Any) -> None:
        if self._truoc_do is not None:
            signal.signal(signal.SIGINT, self._truoc_do)


def uoc_luong_dung_luong_gb(n_params: int, max_keep: int = TRAIN.max_checkpoints_keep) -> float:
    """Đĩa cần cho checkpoint: trọng số fp32 + hai trạng thái Adam = 12 byte/tham số."""
    bytes_moi_ckpt = n_params * 12
    # +1 cho best.pt
    return bytes_moi_ckpt * (max_keep + 1) / 1024**3


def make_manager(config: TrainConfig = TRAIN, directory: Path | None = None) -> CheckpointManager:
    from luna_zero.config import CHECKPOINT_DIR

    return CheckpointManager(directory or CHECKPOINT_DIR, max_keep=config.max_checkpoints_keep)
