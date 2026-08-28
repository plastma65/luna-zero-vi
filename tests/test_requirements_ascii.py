"""requirements.txt phải là ASCII thuần — nếu không pip chết trên Windows.

pip đọc requirements.txt bằng encoding mặc định của hệ thống chứ không phải UTF-8.
Trên Windows tiếng Anh đó là cp1252, và một dấu tiếng Việt trong comment đủ để
`pip install -r requirements.txt` chết với UnicodeDecodeError.

Lỗi này đã xảy ra thật. Nó lọt qua cả pytest lẫn CI vì Linux dùng locale UTF-8 —
đúng kiểu lỗi chỉ lộ trên máy người dùng. Test này đọc file theo đúng cách pip đọc.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Các file được công cụ bên ngoài đọc bằng locale encoding, không phải UTF-8.
FILE_PHAI_ASCII = ["requirements.txt"]


@pytest.mark.parametrize("name", FILE_PHAI_ASCII)
def test_doc_duoc_bang_cp1252(name: str) -> None:
    """Mô phỏng đúng cách pip đọc file trên Windows tiếng Anh."""
    raw = (REPO_ROOT / name).read_bytes()
    try:
        raw.decode("cp1252")
    except UnicodeDecodeError as exc:
        vi_tri = exc.start
        pytest.fail(
            f"{name} có byte không đọc được bằng cp1252 ở vị trí {vi_tri} "
            f"({raw[vi_tri : vi_tri + 8]!r}). pip trên Windows sẽ chết. "
            "Bỏ dấu tiếng Việt trong file này."
        )


@pytest.mark.parametrize("name", FILE_PHAI_ASCII)
def test_la_ascii_thuan(name: str) -> None:
    """Chặt hơn cp1252: ASCII thuần thì mọi locale đều đọc được, không riêng gì Windows."""
    raw = (REPO_ROOT / name).read_bytes()
    khong_ascii = sorted({b for b in raw if b > 127})
    assert not khong_ascii, f"{name} có byte ngoài ASCII: {khong_ascii}"
