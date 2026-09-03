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
  giao_dien/    chat cục bộ: may_chu.py (http.server, stream NDJSON) + trang.html
scripts/
  download_corpus.py  tải corpus HF theo luồng
  train_tokenizer.py  train BPE 32k (có --smoke)
  do_nen.py           đo ký tự/token và so với mức ký tự
  sinh_thu.py         sinh văn bản từ checkpoint ra terminal
  giao_dien.py        mở giao diện chat (có --gia-lap khi chưa có checkpoint)
tests/                round-trip, nén, ô nhiễm, hằng số, VRAM, smoke import
```

## Quyết định kỹ thuật đã chốt (đừng đảo ngược mà không hỏi)

* **BPE mức byte, không SentencePiece.** Lý do: không có UNK nên round-trip luôn đúng
  nguyên văn; không có normalizer ngầm nuốt khoảng trắng.
* **NFC chuẩn hoá ở tầng dữ liệu (`data.normalize_text`), không ở tokenizer.** Nếu nhét
  normalizer vào tokenizer thì tokenizer hết lossless và test round-trip mất giá trị.
* **`add_prefix_space=False`.** Bật lên là hỏng round-trip ngay.
* **Không có token UNK trong `SPECIAL_TOKENS`.** Có UNK chỉ tạo chỗ mất dữ liệu âm thầm.
* **Luật lấy mẫu chỉ có MỘT bản: `model.sinh_dan()`.** `sinh()` chỉ là vỏ gom kết
  quả cho CLI, giao diện web thì lặp thẳng trên generator. Chép luật ra bản thứ hai
  là lỗi số 6 dưới lớp vỏ mới, và vì nó nằm trong xác suất nên chỉ lộ ra dưới dạng
  "giao diện viết khác terminal". `tests/test_giao_dien.py` khoá hai lối gọi lại
  với nhau bằng cùng seed.
* **Stream phải giải mã theo TIỀN TỐ, không theo từng token.** BPE mức byte cắt chữ
  có dấu thành nhiều byte; decode token lẻ ra "\ufffd" giữa câu, trông như model
  viết bậy. Xem `GiaiMaDan`.
* **KHÔNG viết cứng `.cuda()` hay `device="cuda"` ở bất kỳ đâu.** Chọn thiết bị qua một
  chỗ duy nhất (`config.chon_thiet_bi()`), mặc định tự dò, ghi đè được bằng `--device`.

  Lý do: train thì bắt buộc CUDA (CPU chậm hơn 50-100 lần, 4 ngày thành 6-12 tháng),
  nhưng **chạy model thì phải sống được trên máy không GPU**. Đây là điểm Luna Zero hơn
  hẳn Luna cũ: Qwen3-4B cần ~8GB trọng số và `bitsandbytes` vốn chỉ có nhân CUDA, nên
  bỏ GPU ra là không nạp nổi. Luna Zero 110M chỉ 440MB fp32 (~110MB nếu int8), nằm gọn
  trong RAM máy thường và không cần thư viện lượng tử hoá đặc biệt nào.

  Có test bắt buộc model chạy được trên CPU, và test đó chạy trong CI GitHub — nơi
  không có GPU — nên lỗi viết cứng cuda không thể lọt qua.

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

## Họ lỗi nguy hiểm nhất của dự án này: PHÉP ĐO TỰ LỪA

Đã xuất hiện **bốn lần** dưới bốn lớp vỏ khác nhau. Mỗi lần đều cho con số ĐẸP HƠN sự
thật, và không lần nào ném lỗi hay cảnh báo. Đây không phải bốn sự cố rời rạc mà là một
họ lỗi, nên khi thêm bất kỳ phép đo mới nào phải hỏi trước: *dữ liệu đo có dính vào thứ
đang được đo không?*

1. **Luna cũ**: 8/10 câu eval nằm nguyên văn trong data train -> eval đo trí nhớ, và
   bỏ lọt một bước lùi thật.
2. **`do_nen.py`**: bỏ qua 500MB theo hằng số mặc định trong khi tokenizer đã train trên
   2GB -> 1,5GB dữ liệu train lọt vào mẫu đo. Vá bằng `luna_zero_bpe.meta.json` ghi
   `train_bytes`, và in CẢNH BÁO khi thiếu metadata thay vì im lặng dùng mặc định.
3. **`test_loss_ban_dau_bang_ln_vocab`**: truyền `targets=x`, tức bắt model dự đoán chính
   token đang nhìn. Embedding buộc chung làm `logits = h @ E^T`, tích vô hướng của một
   embedding với chính nó lớn hơn với embedding khác, nên loss ra 3,87 thay vì 4,57.
   Test sai, model đúng.
4. **`train.bin` và tokenizer lệch nhau**: `meta.json` chỉ ghi `vocab_size`, mà hai
   tokenizer khác nhau vẫn cùng vocab 32.000. Train sẽ chạy đủ 4 ngày với loss đẹp rồi
   sinh ra chữ rác. Vá bằng vân tay băm nguyên file tokenizer + `kiem_tokenizer_khop()`
   gọi ngay lúc khởi động vòng lặp train.

**Quy tắc rút ra**: mọi phép đo phải nói rõ NÓ ĐO TRÊN DỮ LIỆU NÀO và chứng minh dữ liệu
đó tách khỏi thứ đang được đo. Nếu không chứng minh được thì phép đo đó chưa dùng được.

**Hệ quả cho lúc train thật**: loss khởi đầu phải quanh `ln(32000) = 10,37`. Thấp hơn rõ
rệt KHÔNG phải tin vui — hãy loại trừ nhãn trùng đầu vào và rò rỉ nhân quả trước đã.
