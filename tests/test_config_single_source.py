"""Chặn việc khai lại hằng số cấu hình ở nơi khác ngoài config.py.

Bài học Luna cũ: system prompt tồn tại 3 bản ở 3 script rồi lệch nhau lúc nào không
hay. Test này quét toàn bộ mã nguồn tìm các con số cấu hình viết cứng và bắt đỏ.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from luna_zero import config

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ["src", "scripts"]

# Các giá trị chỉ được phép xuất hiện trong config.py.
CAM_SO = {
    32000: "TOKENIZER.vocab_size / MODEL.vocab_size",
    2200000000: "TRAIN.target_tokens",
    768: "MODEL.d_model",
    1024: "MODEL.block_size",
}
# 1024 cũng là hằng đổi đơn vị byte (1024**2, 1024**3). Chỉ bắt khi nó đứng một mình.
LUY_THUA_BYTE_OK = {1024}

MIEN_TRU = {"config.py"}


def _python_files() -> list[Path]:
    files: list[Path] = []
    for d in SCAN_DIRS:
        files.extend(p for p in (REPO_ROOT / d).rglob("*.py") if p.name not in MIEN_TRU)
    return files


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: p.name)
def test_khong_viet_cung_hang_so(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]

    vi_pham = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, int)):
            continue
        if node.value not in CAM_SO:
            continue
        if node.value in LUY_THUA_BYTE_OK and isinstance(getattr(node, "parent", None), ast.BinOp):
            continue
        vi_pham.append(f"dòng {node.lineno}: {node.value} -> dùng config.{CAM_SO[node.value]}")
    assert not vi_pham, f"{path.relative_to(REPO_ROOT)} khai lại hằng số:\n" + "\n".join(vi_pham)


def test_config_khong_tu_mau_thuan() -> None:
    assert config.MODEL.vocab_size == config.TOKENIZER.vocab_size
    assert config.MODEL.d_model % config.MODEL.n_head == 0
    assert config.TRAIN.max_checkpoints_keep > 0, "phải có trần checkpoint, xem bài học Luna cũ"
