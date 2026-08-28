# Luna Zero

Mô hình ngôn ngữ tiếng Việt **train từ số 0**. Không fine-tune model của ai cả — mọi
con số trong file trọng số đều do dự án này train ra.

Nhánh song song với [Luna](https://github.com/plastma65/luna-vi-companion) (fine-tune
Qwen3-4B bằng QLoRA). Luna Zero không thay thế Luna.

## Quy mô đã chốt

| | |
|---|---|
| Tham số | ~110M (đo bằng `luna_zero.model.estimate_num_params`) |
| Kiến trúc | 12 lớp · d_model 768 · 12 head · block_size 1024 |
| Từ vựng | BPE mức byte, 32.000 |
| Dữ liệu train | 2,2 tỷ token (~6GB văn bản tiếng Việt) |
| VRAM ước tính | ~4,5GB — vừa RTX 3060 12GB |
| Thời gian | 3-5 ngày chạy liên tục |

Đã cân nhắc và loại: 336M (4-7 tuần), 750M (không đủ VRAM), 4B (không khả thi).

## Kỳ vọng thực tế

Ở quy mô 110M, model sẽ **viết tiếng Việt trôi chảy** nhưng **bịa kiến thức rất nhiều**,
**không làm được toán**, và **không theo được chỉ dẫn phức tạp**. Đó là trần của quy mô
này, không phải lỗi huấn luyện.

## Trạng thái

- [x] **Chặng 1 — Tokenizer**: khung dự án, BPE mức byte 32k, đo nén, tầng checkpoint, bộ test
- [ ] Chặng 2 — Dữ liệu: tải 6GB, khử trùng lặp, token hoá ra `.bin`
- [ ] Chặng 3 — Model + vòng lặp train
- [ ] Chặng 4 — Eval và sinh văn bản

## Bắt đầu

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

# 1. Tải corpus (wikipedia không cần đăng nhập; culturax cần chấp nhận điều khoản trên HF)
python scripts/download_corpus.py --source wikipedia --target-gb 1.5

# 2. Chạy thử tí hon trước khi đốt CPU thật
python scripts/train_tokenizer.py --smoke --corpus-dir tests/fixtures

# 3. Train tokenizer thật (~500MB text, vài chục phút CPU)
python scripts/train_tokenizer.py

# 4. Đo tỷ lệ nén — mục tiêu 2,5-3,0 ký tự/token
python scripts/do_nen.py
```

## Ngắt giữa chừng rồi chạy tiếp

Train được thiết kế để **ngắt bất cứ lúc nào**. Nhấn `Ctrl+C` một lần: vòng lặp kết thúc
bước đang chạy, lưu checkpoint rồi thoát. Chạy lại đúng lệnh cũ là nó tự tìm checkpoint
mới nhất và tiếp tục đúng chỗ đã dừng — cả optimizer, scheduler lẫn vị trí trong corpus.

Checkpoint lưu tự động mỗi `TRAIN.save_every_steps` bước (~36 phút), giữ 3 bản mới nhất
cộng `best.pt`, tốn khoảng 5,3GB đĩa. Mất điện đột ngột thì mất tối đa phần công sức
kể từ lần lưu gần nhất.

**Quan trọng trên Windows**: tắt Sleep và Hibernate trong Power Options. Màn hình tắt thì
không sao, nhưng máy ngủ sẽ huỷ CUDA context và tiến trình chết giữa chừng.

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 15   # màn hình vẫn tắt được, không ảnh hưởng GPU
```

## Kiểm tra

```bash
ruff check . && black --check . && pytest -q
```

## Cái gì nằm ở đâu

Repo này **chỉ chứa code và tài liệu**. Mọi thứ nặng hoặc tái tạo được đều nằm ngoài:

| Thứ | Ở đâu | Vì sao |
|---|---|---|
| Code, test, tài liệu | GitHub (repo này) | nhẹ, cần lịch sử thay đổi |
| `artifacts/tokenizer/*.json` | GitHub | ~2MB, là *kết quả* của Chặng 1 và cần bản đúng để tái lập |
| Corpus `data/raw/*.jsonl` | máy local | vài GB, tải lại được bằng `scripts/download_corpus.py` |
| Token đã đóng gói `*.bin` | máy local | sinh lại từ corpus + tokenizer |
| Checkpoint `artifacts/checkpoints/` | máy local | ~1,3GB mỗi bản |
| **Trọng số cuối** | **Hugging Face Hub** | git không hợp để chứa file mô hình |

Đừng dùng Git LFS cho checkpoint. LFS tính dung lượng theo *mọi phiên bản từng đẩy lên*,
nên vài lần push checkpoint 1,3GB là hết hạn mức miễn phí và không xoá lùi được dễ dàng.
Hugging Face Hub miễn phí cho repo model công khai và sinh ra để làm đúng việc này.

`tests/test_gitignore.py` hỏi thẳng `git check-ignore` theo cả hai chiều — thứ phải chặn
và thứ cấm chặn — nên một lần sửa `.gitignore` làm rơi mất dữ liệu quý sẽ đỏ ngay.

## Nguồn dữ liệu

| Nguồn | Cỡ | Gate |
|---|---|---|
| `wikimedia/wikipedia` `20231101.vi` | ~1,5GB | không |
| `uonlp/CulturaX` subset `vi` | 55B token | **có** — phải đồng ý điều khoản + `huggingface-cli login` |
| `oscar-corpus/OSCAR-2301` vi | 68GB | có |
| NewsCorpus (binhvq) | 53GB | không |

Chỉ cần ~6GB, nên dữ liệu dư thừa nhiều lần.

## Giấy phép

[Apache License 2.0](LICENSE) — Copyright 2026 Lozens.
