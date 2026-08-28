"""Round-trip: encode rồi decode phải ra ĐÚNG NGUYÊN VĂN chuỗi gốc.

Đây là phép đo chiều ngược: thay vì kiểm "tokenizer có tách từ đẹp không" (chủ quan,
dễ đỏ oan), ta kiểm thứ tokenizer KHÔNG ĐƯỢC PHÉP làm — làm mất dù chỉ một ký tự.
Không có cách nào lách được phép đo này.
"""

from __future__ import annotations

import pytest

from luna_zero import config

TRICKY = [
    "Xin chào, tôi là Luna.",
    "Nghiêng nghiêng bóng nước Hồ Gươm buổi sớm.",
    # Chữ hiếm gặp trong corpus nhỏ: ép tokenizer rơi về mức byte.
    "Quỳnh Nhuyễn Ngoẵng Khuỵu Tuyệt",
    # Khoảng trắng đầu/cuối và lặp — chỗ SentencePiece hay nuốt mất.
    "  hai   khoảng   trắng  ",
    "\ttab\tvà\nxuống dòng\n\n",
    # Ký tự ngoài tiếng Việt: emoji, chữ Hán, dấu tiền tệ.
    "Giá 100.000₫ 😀 rất rẻ 汉字 mixed",
    "",
    "a",
    "0123456789 !@#$%^&*()_+-=[]{}|;':\",./<>?",
]


@pytest.mark.parametrize("text", TRICKY)
def test_roundtrip_nguyen_van(tiny_tokenizer, text: str) -> None:
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text


def test_roundtrip_tren_toan_bo_corpus_mau(tiny_tokenizer, sample_texts) -> None:
    for text in sample_texts:
        assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text


def test_khong_co_token_unk(tiny_tokenizer) -> None:
    """Không UNK là điều kiện cần để round-trip đúng với mọi đầu vào."""
    vocab_words = {"<unk>", "[UNK]", "<UNK>"}
    ids = tiny_tokenizer.encode("Ω≈ç√∫˜µ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢")
    assert ids, "chuỗi lạ vẫn phải mã hoá được"
    assert tiny_tokenizer.decode(ids) == "Ω≈ç√∫˜µ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢"
    decoded_vocab = {tiny_tokenizer.decode([i]) for i in range(min(300, tiny_tokenizer.vocab_size))}
    assert not (decoded_vocab & vocab_words)


def test_encode_document_bao_bos_eos(tiny_tokenizer) -> None:
    ids = tiny_tokenizer.encode_document("Một câu.")
    assert ids[0] == config.BOS_ID
    assert ids[-1] == config.EOS_ID
    assert len(ids) == len(tiny_tokenizer.encode("Một câu.")) + 2
