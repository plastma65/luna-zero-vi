"""BPE mức byte cho tiếng Việt: train, nạp, và đo tỷ lệ nén."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from luna_zero import config


def build_empty_tokenizer() -> Tokenizer:
    """Dựng tokenizer BPE mức byte chưa train.

    Chọn byte-level BPE (không phải SentencePiece unigram) vì hai lý do:
    1. Không có UNK. Mọi chuỗi UTF-8 đều mã hoá được, nên round-trip luôn đúng nguyên
       văn — kể cả emoji, chữ Hán lọt vào corpus, hay dấu tiếng Việt lạ.
    2. Không có normalizer ngầm. SentencePiece mặc định NFKC hoá và nuốt khoảng trắng,
       làm giải mã ra chuỗi khác chuỗi gốc. Chuẩn hoá đã làm ở data.normalize_text.
    """
    tok = Tokenizer(models.BPE(unk_token=None))
    # add_prefix_space=False: không tự chèn khoảng trắng đầu chuỗi, nếu không thì
    # decode(encode(x)) trả về " " + x và round-trip hỏng.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


def train_tokenizer(
    corpus: Iterable[str],
    output_path: Path,
    vocab_size: int = config.TOKENIZER.vocab_size,
    min_frequency: int = config.TOKENIZER.min_frequency,
) -> Tokenizer:
    """Train BPE trên một iterable các document rồi lưu ra `output_path`."""
    tok = build_empty_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        # Special token thêm trước tiên nên chiếm id 0,1,2 đúng như config khai báo.
        special_tokens=list(config.SPECIAL_TOKENS),
        # Ép đủ 256 ký tự byte vào bảng chữ cái ban đầu: đây là thứ bảo đảm không UNK.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(corpus, trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_path))
    return tok


class LunaTokenizer:
    """Lớp bọc mỏng quanh `tokenizers.Tokenizer`, gắn sẵn quy ước id đặc biệt."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer

    @classmethod
    def load(cls, path: Path | None = None) -> LunaTokenizer:
        path = path or config.TOKENIZER_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Chưa có tokenizer ở {path}. Chạy: python scripts/train_tokenizer.py"
            )
        return cls(Tokenizer.from_file(str(path)))

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        # skip_special_tokens=False để round-trip kiểm được đúng nguyên văn.
        return self._tok.decode(ids, skip_special_tokens=False)

    def encode_document(self, text: str) -> list[int]:
        """Mã hoá một document cho việc train: <bos> ... <eos>."""
        return [config.BOS_ID, *self.encode(text), config.EOS_ID]


@dataclass(frozen=True)
class CompressionStats:
    n_chars: int
    n_tokens: int
    n_docs: int

    @property
    def chars_per_token(self) -> float:
        return self.n_chars / self.n_tokens if self.n_tokens else 0.0


def measure_compression(tokenizer: LunaTokenizer, texts: Iterable[str]) -> CompressionStats:
    """Đếm ký tự/token trên một mẫu văn bản.

    Đo bằng KÝ TỰ Unicode, không phải byte: tiếng Việt có dấu chiếm 2-3 byte/ký tự nên
    đếm byte sẽ thổi phồng con số nén lên gấp đôi và tự lừa mình.
    """
    n_chars = n_tokens = n_docs = 0
    for text in texts:
        n_chars += len(text)
        n_tokens += len(tokenizer.encode(text))
        n_docs += 1
    return CompressionStats(n_chars=n_chars, n_tokens=n_tokens, n_docs=n_docs)


def char_level_baseline(texts: Iterable[str]) -> CompressionStats:
    """Mốc so sánh: tokenizer mức ký tự luôn cho đúng 1,0 ký tự/token."""
    n_chars = n_docs = 0
    for text in texts:
        n_chars += len(text)
        n_docs += 1
    return CompressionStats(n_chars=n_chars, n_tokens=n_chars, n_docs=n_docs)
