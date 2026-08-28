# Luna Zero — hướng dẫn cho trợ lý AI

## Dự án là gì
Train mô hình ngôn ngữ tiếng Việt 110M tham số **từ số 0**. Không fine-tune, không nạp
trọng số của ai. Nếu một đề xuất dẫn tới việc khởi tạo từ checkpoint có sẵn, nó sai
mục tiêu dự án — nói thẳng ra thay vì làm.

Dự án cũ `D:\Luna_Project` (fine-tune Qwen3-4B, QLoRA) là nhánh khác. **Không đụng vào.**

## Cách làm việc
1. Trả lời bằng **tiếng Việt**.
2. Sửa thẳng vào code. Không viết guide bắt người dùng chép tay.
3. Comment tiếng Việt ở chỗ khó, giải thích **vì sao chọn cách này**, không mô tả lại code.
4. Trước khi báo xong, tự chạy `ruff check . && black --check . && pytest -q` và thấy xanh.
5. Code nặng phải có bản smoke chạy 1-2 bước trên dữ liệu tí hon trước.
6. Thấy lỗi ngoài phạm vi được nhờ thì nói thẳng, kèm mức nghiêm trọng.
7. Ưu tiên đơn giản, ít phụ thuộc.

Người dùng đã biết Python và ML cơ bản (đã tự viết mini-GPT, đã fine-tune QLoRA, đã
dựng RAG và bộ eval). Không giải thích lại từ vỡ lòng.

## Sáu điều KHÔNG được lặp lại (rút từ dự án Luna cũ, mỗi cái đã gây thiệt hại thật)

1. **`.gitignore` chặn nhầm dữ liệu quý.** `data/processed/*` từng làm 176 mẫu dataset
   không bao giờ được commit. → Chặn theo **loại và kích thước** (`*.bin`, `*.pt`,
   `*.jsonl`, `.venv/`, checkpoint), không chặn cả thư mục dữ liệu. Text nhỏ thì cho lên git.
2. **Checkpoint không giới hạn.** Từng ngốn 1,4GB. → Luôn có `max_checkpoints_keep`.
3. **Eval trùng dữ liệu train.** 8/10 câu eval nằm nguyên văn trong data → eval đo trí
   nhớ chứ không đo hành vi. → Có test tự động chặn ô nhiễm
   (`tests/test_compression.py::test_fixture_khong_lan_vao_corpus_train`).
4. **Kiểm kết quả bằng regex bám câu chữ.** Đỏ oan 4 lần. → Kiểm **hành vi**, và ưu tiên
   **phép đo chiều ngược** (thứ model/code KHÔNG được làm) vì nó không lách được.
   Ví dụ trong repo: round-trip tokenizer, `test_vram_phan_ung_dung_chieu_voi_block_size`.
5. **Test viết xong phải thử phá.** Viết test rồi cố tình tạo đúng loại lỗi nó sinh ra
   để bắt, xem test có đỏ thật không. (Đã làm với `add_prefix_space`, `initial_alphabet`,
   hằng số viết cứng, `block_size`, NFC.)
6. **Cấu hình chép nhiều bản rồi lệch nhau.** → Mọi hằng số ở `src/luna_zero/config.py`.
   `tests/test_config_single_source.py` quét AST của `src/` và `scripts/` và bắt đỏ nếu
   thấy 32000 / 768 / 1024 / 2_200_000_000 viết cứng ở nơi khác.

## Bố cục

```
src/luna_zero/
  config.py     nguồn hằng số DUY NHẤT (đường dẫn, TokenizerConfig, ModelConfig, TrainConfig)
  data.py       chuẩn hoá NFC, đọc/ghi .jsonl, duyệt corpus
  tokenizer.py  BPE mức byte: train, LunaTokenizer, đo nén
  model.py      đếm tham số + ước lượng VRAM (mạng thật ở Chặng 3)
  checkpoint.py lưu/khôi phục trạng thái train, xoay vòng, Ctrl+C mềm
  train.py      lập kế hoạch + tính phần còn lại (vòng lặp thật ở Chặng 3)
scripts/
  download_corpus.py  tải corpus HF theo luồng
  train_tokenizer.py  train BPE 32k (có --smoke)
  do_nen.py           đo ký tự/token và so với mức ký tự
tests/                round-trip, nén, ô nhiễm, hằng số, VRAM, smoke import
```

## Quyết định kỹ thuật đã chốt (đừng đảo ngược mà không hỏi)

* **BPE mức byte, không SentencePiece.** Lý do: không có UNK nên round-trip luôn đúng
  nguyên văn; không có normalizer ngầm nuốt khoảng trắng.
* **NFC chuẩn hoá ở tầng dữ liệu (`data.normalize_text`), không ở tokenizer.** Nếu nhét
  normalizer vào tokenizer thì tokenizer hết lossless và test round-trip mất giá trị.
* **`add_prefix_space=False`.** Bật lên là hỏng round-trip ngay.
* **Không có token UNK trong `SPECIAL_TOKENS`.** Có UNK chỉ tạo chỗ mất dữ liệu âm thầm.

## Checkpoint và chạy tiếp (Chặng 1 đã dựng xong tầng này)

Máy không chạy liên tục 4 ngày được, nên vòng lặp train ở Chặng 3 **bắt buộc** dùng
`CheckpointManager`. Bốn quy tắc, mỗi cái đã có test và đã thử phá:

1. Checkpoint phải chứa **model + optimizer + scheduler + `data_position` + RNG state**.
   Thiếu optimizer thì loss vọt lên khi chạy tiếp; thiếu `data_position` thì mô hình đọc
   lại từ đầu corpus mãi mãi. Cả hai đều **không ném lỗi** — đó là lý do phải có test.
2. Ghi ra `.tmp` rồi `os.replace`. Không bao giờ ghi thẳng lên file cuối.
3. `load_latest()` lùi về bản cũ hơn nếu bản mới nhất hỏng. Đừng bỏ cơ chế này.
4. Xoay vòng theo `TRAIN.max_checkpoints_keep`; `best.pt` không bị xoay vòng đụng tới.

Ctrl+C dùng `NgatMemMai`: lần một đặt cờ để vòng lặp lưu xong rồi thoát, lần hai thoát ngay.
