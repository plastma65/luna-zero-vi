"""Không được viết cứng cuda ở đâu ngoài config.chon_thiet_bi().

Train bắt buộc GPU, nhưng model phải CHẠY được trên máy không GPU — đó là điểm Luna Zero
hơn hẳn Luna cũ (Qwen3-4B + bitsandbytes chỉ có nhân CUDA nên bỏ GPU ra là không nạp nổi).
Một dòng `.cuda()` viết cứng đủ để đánh mất điều đó, và nó chỉ lộ ra khi ai đó thử chạy
trên máy khác — tức là quá muộn.

Test này KHÔNG cần torch nên nó chạy được ở mọi nơi, kể cả CI không có GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# config.py được phép: đó là chỗ duy nhất quyết định thiết bị.
MIEN_TRU = {"config.py"}
# Mẫu đầu tiên mình viết chỉ bắt `device="cuda"` và đã BỎ LỌT `device_type="cuda"`
# ngay trong train.py. Bắt cả cụm gán chuỗi "cuda" cho bất kỳ tham số nào.
CAM = (
    ".cuda(",
    'device="cuda"',
    "device='cuda'",
    'device_type="cuda"',
    "device_type='cuda'",
    'to("cuda")',
    "to('cuda')",
)


def _files() -> list[Path]:
    out: list[Path] = []
    for d in ("src", "scripts"):
        out += [p for p in (REPO_ROOT / d).rglob("*.py") if p.name not in MIEN_TRU]
    return out


@pytest.mark.parametrize("path", _files(), ids=lambda p: p.name)
def test_khong_co_cuda_viet_cung(path: Path) -> None:
    noi_dung = path.read_text(encoding="utf-8")
    # Bỏ dòng chú thích: nhắc tới cuda trong comment là bình thường và cần thiết.
    ma = "\n".join(line for line in noi_dung.splitlines() if not line.lstrip().startswith("#"))
    vi_pham = [c for c in CAM if c in ma]
    assert not vi_pham, (
        f"{path.relative_to(REPO_ROOT)} viết cứng {vi_pham}. "
        "Dùng config.chon_thiet_bi() rồi truyền device xuống."
    )


@pytest.mark.parametrize("path", _files(), ids=lambda p: p.name)
def test_moi_chuoi_cuda_deu_la_so_sanh_chu_khong_phai_gan(path: Path) -> None:
    """Lưới rộng hơn danh sách CAM: mọi lần xuất hiện chuỗi "cuda" trong mã (không phải
    chú thích) phải nằm trong một phép SO SÁNH hoặc startswith, không phải phép gán."""
    ma = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if "cuda" in line and not line.lstrip().startswith("#")
    ]
    xau = [
        line.strip()
        for line in ma
        # help= và docstring được phép nhắc tên thiết bị: đó là văn bản cho người đọc,
        # không phải phép gán. Bản đầu của test này đỏ oan vì quên hai trường hợp đó —
        # test bắt nhầm code đúng cũng tai hại ngang test bỏ lọt code sai.
        if not any(
            k in line for k in ("startswith", "==", "!=", "if ", "is_available", "help=", '"""')
        )
    ]
    assert not xau, f"{path.relative_to(REPO_ROOT)} gán cứng cuda: {xau}"


def test_chon_thiet_bi_ton_tai_va_nhan_ghi_de() -> None:
    """Phải ghi đè được, nếu không thì `--device cpu` vô nghĩa."""
    src = (REPO_ROOT / "src" / "luna_zero" / "config.py").read_text(encoding="utf-8")
    cay = ast.parse(src)
    ham = [n for n in ast.walk(cay) if isinstance(n, ast.FunctionDef)]
    ten = {f.name for f in ham}
    assert "chon_thiet_bi" in ten
    f = next(x for x in ham if x.name == "chon_thiet_bi")
    assert [a.arg for a in f.args.args] == ["uu_tien"]
