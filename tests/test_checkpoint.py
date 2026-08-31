"""Ngắt giữa chừng rồi chạy tiếp — phần dễ hỏng im lặng nhất của một lần train dài."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pytest

from luna_zero.checkpoint import (
    BEST_NAME,
    CheckpointManager,
    NgatMemMai,
    TrainState,
    uoc_luong_dung_luong_gb,
)


def _pickle_save(obj: Any, path: Path) -> None:
    path.write_bytes(pickle.dumps(obj))


def _pickle_load(path: Path) -> Any:
    return pickle.loads(path.read_bytes())


@pytest.fixture
def manager(tmp_path: Path) -> CheckpointManager:
    return CheckpointManager(tmp_path, max_keep=3, save_fn=_pickle_save, load_fn=_pickle_load)


def _state(step: int) -> TrainState:
    return TrainState(step=step, tokens_seen=step * 65_536, data_position=step * 65_536)


def _payload(step: int) -> dict[str, Any]:
    return {"model": f"w{step}", "optimizer": f"adam{step}", "rng": f"r{step}"}


# --- chạy tiếp đúng chỗ ----------------------------------------------------
def test_chua_co_gi_thi_tra_ve_none(manager: CheckpointManager) -> None:
    assert manager.load_latest() is None


def test_luu_roi_nap_lai_dung_nguyen_trang_thai(manager: CheckpointManager) -> None:
    manager.save(_state(200), _payload(200))
    loaded = manager.load_latest()
    assert loaded is not None
    state, payload = loaded
    assert state == _state(200)
    assert payload == _payload(200)


def test_nap_ban_moi_nhat_chu_khong_phai_ban_dau(manager: CheckpointManager) -> None:
    for step in (200, 400, 600):
        manager.save(_state(step), _payload(step))
    state, payload = manager.load_latest()
    assert state.step == 600
    assert payload["model"] == "w600"


def test_optimizer_va_vi_tri_du_lieu_song_sot(manager: CheckpointManager) -> None:
    """Phép đo chiều ngược: checkpoint KHÔNG được phép đánh rơi hai thứ này.

    Mất optimizer -> loss vọt lên khi chạy tiếp. Mất data_position -> mô hình đọc lại
    từ đầu corpus và không bao giờ thấy phần cuối. Cả hai đều không ném lỗi.
    """
    manager.save(TrainState(step=1000, tokens_seen=999, data_position=123_456), _payload(1000))
    state, payload = manager.load_latest()
    assert state.data_position == 123_456
    assert "optimizer" in payload and payload["optimizer"] == "adam1000"


# --- xoay vòng, không để phình đĩa -----------------------------------------
def test_giu_dung_max_keep_ban(manager: CheckpointManager) -> None:
    for step in range(200, 1400, 200):
        manager.save(_state(step), _payload(step))
    con_lai = sorted(p.name for p in manager.directory.glob("ckpt_step_*.pt"))
    assert len(con_lai) == 3
    assert con_lai == [
        "ckpt_step_000000800.pt",
        "ckpt_step_000001000.pt",
        "ckpt_step_000001200.pt",
    ]


def test_best_khong_bi_xoay_vong_xoa(manager: CheckpointManager) -> None:
    manager.save(_state(200), _payload(200), is_best=True)
    for step in range(400, 1600, 200):
        manager.save(_state(step), _payload(step))
    assert (manager.directory / BEST_NAME).exists()
    assert _pickle_load(manager.directory / BEST_NAME)["model"] == "w200"


def test_max_keep_khong_hop_le_bi_chan_som(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CheckpointManager(tmp_path, max_keep=0)


# --- hỏng giữa chừng -------------------------------------------------------
def test_ghi_that_bai_khong_lam_mat_ban_cu(tmp_path: Path) -> None:
    """Cúp điện đúng lúc đang ghi: bản cũ phải còn nguyên và nạp lại được."""
    calls = {"n": 0}

    def save_hong_lan_hai(obj: Any, path: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            path.write_bytes(b"\x80rac-do-dang")  # ghi được một nửa rồi chết
            raise OSError("mất điện")
        _pickle_save(obj, path)

    m = CheckpointManager(tmp_path, max_keep=3, save_fn=save_hong_lan_hai, load_fn=_pickle_load)
    m.save(_state(200), _payload(200))
    with pytest.raises(OSError):
        m.save(_state(400), _payload(400))

    assert not list(tmp_path.glob("*.tmp")), "file dở dang phải được dọn"
    # Điểm mấu chốt của ghi nguyên tử: file đích của bước hỏng KHÔNG ĐƯỢC tồn tại.
    # Nếu nó nằm lại dưới dạng rác, một lần dựng lại manifest sau này sẽ nạp phải nó.
    assert not (tmp_path / "ckpt_step_000000400.pt").exists(), "rác dở dang còn nằm trên đĩa"
    state, payload = m.load_latest()
    assert state.step == 200, "phải quay về bản tốt cuối cùng, không phải bản hỏng"
    assert payload["model"] == "w200"


def test_manifest_hong_van_chay_tiep_duoc(manager: CheckpointManager) -> None:
    """Manifest chỉ là chỉ mục. Hỏng nó không được phép chặn việc chạy tiếp."""
    for step in (200, 400):
        manager.save(_state(step), _payload(step))
    manager.manifest_path.write_text("{ day khong phai json", encoding="utf-8")
    state, _ = manager.load_latest()
    assert state.step == 400


def test_bo_qua_file_da_bi_xoa_tay(manager: CheckpointManager) -> None:
    for step in (200, 400):
        manager.save(_state(step), _payload(step))
    (manager.directory / "ckpt_step_000000400.pt").unlink()
    state, _ = manager.load_latest()
    assert state.step == 200


def test_manifest_la_json_hop_le(manager: CheckpointManager) -> None:
    manager.save(_state(200), _payload(200))
    entries = json.loads(manager.manifest_path.read_text(encoding="utf-8"))
    assert entries[0]["step"] == 200


# --- Ctrl+C mềm ------------------------------------------------------------
def test_ctrl_c_lan_dau_chi_dat_co(capsys) -> None:
    with NgatMemMai() as guard:
        assert not guard.duoc_yeu_cau_dung
        guard._xu_ly(2, None)
        assert guard.duoc_yeu_cau_dung
    assert "lưu checkpoint" in capsys.readouterr().out


def test_ctrl_c_lan_hai_thoat_ngay() -> None:
    with NgatMemMai() as guard:
        guard._xu_ly(2, None)
        with pytest.raises(KeyboardInterrupt):
            guard._xu_ly(2, None)


# --- đĩa -------------------------------------------------------------------
def test_uoc_luong_dung_luong_dia_hop_ly() -> None:
    pytest.importorskip("torch", reason="estimate_num_params nằm trong model.py")
    from luna_zero.model import estimate_num_params

    gb = uoc_luong_dung_luong_gb(estimate_num_params())
    assert 3.0 <= gb <= 8.0, f"ước {gb:.1f}GB — kiểm lại trần checkpoint"


def test_ghi_hong_cong_manifest_hong_van_ve_duoc_ban_tot(tmp_path: Path) -> None:
    """Hai sự cố chồng nhau: ghi dở dang rồi manifest cũng hỏng.

    Đây là chuỗi thật mà lỗ hổng ban đầu để lọt: nếu file rác còn trên đĩa, việc dựng
    lại manifest từ tên file sẽ chọn đúng file rác vì nó có số bước lớn hơn.
    """
    calls = {"n": 0}

    def save_hong_lan_hai(obj: Any, path: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            path.write_bytes(b"\x80rac")
            raise OSError("mất điện")
        _pickle_save(obj, path)

    m = CheckpointManager(tmp_path, max_keep=3, save_fn=save_hong_lan_hai, load_fn=_pickle_load)
    m.save(_state(200), _payload(200))
    with pytest.raises(OSError):
        m.save(_state(400), _payload(400))
    m.manifest_path.write_text("rac", encoding="utf-8")

    state, payload = m.load_latest()
    assert state.step == 200
    assert payload["model"] == "w200"


def test_lui_ve_ban_truoc_khi_ban_moi_nhat_hong(manager: CheckpointManager, capsys) -> None:
    """Phép đo chiều ngược: một checkpoint hỏng KHÔNG được phép giết cả lần train."""
    for step in (200, 400):
        manager.save(_state(step), _payload(step))
    (manager.directory / "ckpt_step_000000400.pt").write_bytes(b"\x80hong")

    state, payload = manager.load_latest()
    assert state.step == 200
    assert payload["model"] == "w200"
    assert "lùi về bản trước" in capsys.readouterr().out
