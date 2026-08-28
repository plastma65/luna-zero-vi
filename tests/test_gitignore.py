"""Kiểm .gitignore bằng chính git, không phải bằng suy đoán.

Bài học Luna cũ nặng nhất: dòng `data/processed/*` khiến 176 mẫu dataset không bao giờ
được commit, suýt mất trắng — và không ai biết vì `git status` im lặng đúng như khi
mọi thứ ổn. Test này hỏi thẳng `git check-ignore` xem từng đường dẫn có bị chặn không,
theo cả hai chiều: thứ PHẢI chặn, và thứ CẤM chặn.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Nặng hoặc tái tạo được -> phải nằm ngoài repo.
PHAI_CHAN = [
    "data/raw/wikipedia_vi_0000.jsonl",
    "data/raw/culturax_vi_0003.jsonl",
    "data/processed/train.bin",
    "data/processed/val.bin",
    "artifacts/checkpoints/ckpt_step_000012000.pt",
    "artifacts/checkpoints/best.pt",
    "artifacts/checkpoints/manifest.json",
    "artifacts/tokenizer/smoke_bpe.json",
    ".venv/Scripts/python.exe",
    "src/luna_zero/__pycache__/config.cpython-311.pyc",
    "model.safetensors",
    "corpus.txt.gz",
]

# Nhỏ và quý -> CẤM chặn. Đây là chiều mà Luna cũ đã sai.
CAM_CHAN = [
    "tests/fixtures/vi_sample.jsonl",
    "artifacts/tokenizer/luna_zero_bpe.json",
    "data/raw/README.md",
    "data/processed/README.md",
    "src/luna_zero/config.py",
    "scripts/train_tokenizer.py",
    "README.md",
    "CLAUDE.md",
    "LICENSE",
    ".github/workflows/ci.yml",
    "pyproject.toml",
    "requirements.txt",
]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="không có git hoặc chưa git init",
)


def _bi_chan(path: str) -> bool:
    # check-ignore trả mã 0 nếu bị chặn, 1 nếu không. --no-index để hỏi được cả
    # đường dẫn chưa tồn tại trên đĩa.
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", path],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return proc.returncode == 0


@pytest.mark.parametrize("path", PHAI_CHAN)
def test_file_nang_bi_chan(path: str) -> None:
    assert _bi_chan(path), f"{path} PHẢI bị .gitignore chặn — đừng để nó vào repo"


@pytest.mark.parametrize("path", CAM_CHAN)
def test_file_quy_khong_bi_chan(path: str) -> None:
    assert not _bi_chan(path), (
        f"{path} đang bị .gitignore chặn nhầm. Đây đúng là lỗi đã làm mất "
        "176 mẫu dataset ở dự án Luna cũ."
    )
