"""File tiến độ trong repo — để biết đang ở bước nào mà không cần mở thư mục checkpoint.

Checkpoint hay được đặt ở ổ khác (SSD cho nhanh). Ai chỉ thấy repo thì mù hoàn toàn
về tiến độ, kể cả chính Claude ở phiên sau. File này vá đúng chỗ đó.
"""

from __future__ import annotations

import json
from pathlib import Path

from luna_zero.checkpoint import TrainState
from luna_zero.train import ghi_tien_do


def _ghi(thu_muc: Path, step: int, loss: float = 3.0, best: float = 3.1) -> None:
    ghi_tien_do(
        thu_muc,
        TrainState(
            step=step, tokens_seen=step * 65_536, data_position=step * 64, best_val_loss=best
        ),
        tong_buoc=33_569,
        loss=loss,
        lr=6e-4,
        tps=23_900,
        vram_gb=6.9,
    )


def test_ghi_duoc_va_doc_lai_dung(tmp_path: Path) -> None:
    _ghi(tmp_path, 12_000)
    d = json.loads((tmp_path / "tien_do.json").read_text(encoding="utf-8"))
    assert d["step"] == 12_000
    assert d["tien_do"] == round(12_000 / 33_569, 4)
    assert d["tokens_per_second"] == 23_900
    assert d["best_val_loss"] == 3.1


def test_ghi_de_ban_moi_khong_de_lai_file_tam(tmp_path: Path) -> None:
    """Ghi nguyên tử: Ctrl+C đúng lúc đang ghi không được để lại .tmp hay file hỏng."""
    _ghi(tmp_path, 100)
    _ghi(tmp_path, 200)
    assert json.loads((tmp_path / "tien_do.json").read_text(encoding="utf-8"))["step"] == 200
    assert not list(tmp_path.glob("*.tmp"))


def test_lich_su_giu_moi_ban_ghi(tmp_path: Path) -> None:
    """tien_do.json là ảnh chụp hiện tại; lich_su_train.jsonl là cả đường loss.

    Phép đo chiều ngược: lịch sử KHÔNG được bị ghi đè — mất nó là mất khả năng vẽ lại
    đường loss ở Chặng 4, và mất luôn bằng chứng model có tiến bộ thật hay không.
    """
    for i, step in enumerate((100, 200, 300)):
        _ghi(tmp_path, step, loss=3.5 - i * 0.1)
    dong = (tmp_path / "lich_su_train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(dong) == 3
    assert [json.loads(x)["step"] for x in dong] == [100, 200, 300]
    assert [json.loads(x)["loss"] for x in dong] == [3.5, 3.4, 3.3]


def test_best_val_vo_han_thi_ghi_null(tmp_path: Path) -> None:
    """Chưa eval lần nào thì best_val_loss là inf — mà inf không phải JSON hợp lệ."""
    ghi_tien_do(
        tmp_path,
        TrainState(step=1, tokens_seen=0, data_position=0),
        tong_buoc=10,
        loss=10.3,
        lr=1e-6,
        tps=0,
        vram_gb=0.0,
    )
    thô = (tmp_path / "tien_do.json").read_text(encoding="utf-8")
    assert "Infinity" not in thô, "JSON có Infinity thì trình phân tích chuẩn sẽ từ chối"
    assert json.loads(thô)["best_val_loss"] is None


def test_tong_buoc_bang_khong_khong_chia_cho_khong(tmp_path: Path) -> None:
    ghi_tien_do(
        tmp_path,
        TrainState(step=0, tokens_seen=0, data_position=0),
        tong_buoc=0,
        loss=0.0,
        lr=0.0,
        tps=0,
        vram_gb=0.0,
    )
    assert json.loads((tmp_path / "tien_do.json").read_text(encoding="utf-8"))["tien_do"] == 0.0
