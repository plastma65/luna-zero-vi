"""Đo chất lượng văn bản sinh ra bằng con số, không bằng cảm nhận.

Bài học Luna cũ số 4: kiểm kết quả model bằng cách bám câu chữ đã đỏ oan bốn lần.
Ở đây cũng vậy — "trông có vẻ ít lặp hơn" không phải kết luận, nó là ấn tượng. Ba chỉ
số dưới đây đều là PHÉP ĐO CHIỀU NGƯỢC: chúng đo thứ model KHÔNG được làm (lặp lại
chính mình), nên không lách được bằng cách viết hay hơn.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

# NEO ĐO ĐƯỢC, không phải con số tra sách: distinct-2 trung vị của 284 document
# tiếng Việt do người viết, lấy từ chính corpus train (culturax_vi_0030).
# Dùng làm mốc hai chiều:
#   thấp hơn nhiều  -> model đang lặp (chưa train đủ, hoặc lấy mẫu quá hẹp)
#   CAO HƠN nhiều   -> phạt lặp quá tay; văn bản thật CÓ lặp, đè hết là thành bất thường
DISTINCT2_NGUOI_VIET = 0.853
DISTINCT2_KHOANG_TU_NHIEN = (0.78, 0.93)


def trong_khoang_tu_nhien(d2: float) -> bool:
    """distinct-2 có nằm trong khoảng văn bản người viết không."""
    thap, cao = DISTINCT2_KHOANG_TU_NHIEN
    return thap <= d2 <= cao


def _ngram(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


@dataclass(frozen=True)
class DoLap:
    distinct_1: float
    distinct_2: float
    distinct_4: float
    lap_dai_nhat: int
    n_token: int

    def __str__(self) -> str:
        return (
            f"distinct-1 {self.distinct_1:.3f} | distinct-2 {self.distinct_2:.3f} "
            f"| distinct-4 {self.distinct_4:.3f} | chuỗi lặp dài nhất {self.lap_dai_nhat}"
        )


def do_lap(van_ban: str) -> DoLap:
    """distinct-n = tỷ lệ n-gram KHÁC NHAU trên tổng số n-gram.

    Càng gần 1 càng ít lặp. Văn bản tiếng Việt tự nhiên thường có distinct-2 quanh
    0,85-0,95; dưới 0,6 là đã lặp nặng. `lap_dai_nhat` bắt kiểu suy thoái tệ nhất:
    một cụm bị nhắc lại nguyên văn nhiều lần liền nhau.
    """
    tokens = van_ban.split()
    if not tokens:
        return DoLap(0.0, 0.0, 0.0, 0, 0)

    def distinct(n: int) -> float:
        g = _ngram(tokens, n)
        return len(set(g)) / len(g) if g else 0.0

    # Cụm 4-từ xuất hiện nhiều lần nhất — con số này bắt đúng vòng lặp mà mắt thấy.
    bon = _ngram(tokens, 4)
    lap = max(Counter(bon).values()) if bon else 0

    return DoLap(
        distinct_1=distinct(1),
        distinct_2=distinct(2),
        distinct_4=distinct(4),
        lap_dai_nhat=lap,
        n_token=len(tokens),
    )
