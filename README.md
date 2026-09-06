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
| Context | 1024 token |
| Trọng số phát hành | ~441MB safetensors, inference-only |

Đã cân nhắc và loại: 336M (4-7 tuần), 750M (không đủ VRAM), 4B (không khả thi).

## Kỳ vọng thực tế

Ở quy mô 110M, model có thể **viết tiếng Việt khá trôi chảy** nhưng **bịa kiến thức rất nhiều**,
**không làm được toán đáng tin cậy**, và **không theo được chỉ dẫn phức tạp**. Không nên
dùng model như một nguồn thông tin thực tế đáng tin cậy.

## Trạng thái

- [x] **Chặng 1 — Tokenizer**: Byte-level BPE 32k, đo nén, checkpoint, test
- [x] **Chặng 2 — Dữ liệu**: tải corpus, dedup, split, token hoá `.bin`
- [x] **Chặng 3 — Model + train**: GPT decoder-only ~110M, train đủ 2,2B token
- [x] **Chặng 4 — Eval và sinh văn bản**: held-out final, generation final, release package

### Ghi chú

- Checkpoint phát hành: **step 33.569**
- Human-authored final held-out: **100 document / 17.580 token**
- Final NLL: **3.2195**
- Final perplexity: **25.016**
- Exact document overlap với raw train corpus: **0 / 2.414.008 document**
- Final generation: **30 mẫu**, distinct-2 mean **0.948**, distinct-4 mean **0.994**
- 4-gram lặp tối đa: **3**
- Exact contamination guard không chứng minh được near-duplicate hoặc paraphrase
- distinct-n chỉ mô tả repetition, **không phải** điểm factuality hay độ “tự nhiên”

Trọng số inference-only:

**[Lozens/Luna-Zero-110M — Hugging Face](https://huggingface.co/Lozens/Luna-Zero-110M)**

## Bắt đầu

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Tải corpus
python scripts/download_corpus.py --source wikipedia --target-gb 1.5

# Smoke tokenizer
python scripts/train_tokenizer.py --smoke --corpus-dir tests/fixtures

# Train tokenizer
python scripts/train_tokenizer.py

# Đo tỷ lệ nén
python scripts/do_nen.py
```

## Nói chuyện với model

```bash
python scripts/giao_dien.py
python scripts/giao_dien.py --gia-lap
python scripts/giao_dien.py --device cpu --port 8080
```

Giao diện chat cục bộ ở `http://127.0.0.1:8765`. Dưới mỗi câu trả lời có metric repetition
như `distinct-2`; metric này không được dùng như điểm “tự nhiên”, factuality hay chất lượng
nội dung tổng quát.

Sampling mặc định của bản phát hành:

```json
{
  "so_token": 120,
  "temperature": 0.9,
  "top_k": null,
  "top_p": 0.92,
  "phat_lap": 1.05
}
```

## Checkpoint và release

Checkpoint train chứa optimizer, RNG state và trạng thái resume nên không đưa lên GitHub
hoặc Hugging Face release package.

Bản Hugging Face chỉ chứa trọng số inference-only `model.safetensors`, tokenizer, config,
model card và artifact eval cần thiết để kiểm chứng release.

Package đã được kiểm theo vòng:

```text
local export -> Hugging Face upload -> download sạch -> CPU load + forward PASS
```

Kiến trúc Luna Zero là custom PyTorch, **không giả vờ tương thích trực tiếp với
`transformers.AutoModel`**. Dùng source code Luna Zero để dựng `LunaZeroGPT`, nạp
`model.safetensors` và tokenizer đi kèm.

## Kiểm tra

```bash
ruff check . && black --check . && pytest -q
```

## Cái gì nằm ở đâu

Repo này **chỉ chứa code và tài liệu**. Mọi thứ nặng hoặc tái tạo được đều nằm ngoài:

| Thứ | Ở đâu | Vì sao |
|---|---|---|
| Code, test, tài liệu | GitHub | nhẹ, cần lịch sử thay đổi |
| Tokenizer artifact | GitHub | cần bản đúng để tái lập |
| Corpus `data/raw/*.jsonl` | máy local | vài GB |
| Token đã đóng gói `*.bin` | máy local | sinh lại được |
| Checkpoint train | máy local | chứa optimizer/RNG/resume state |
| **Trọng số inference-only** | **Hugging Face Hub** | phù hợp cho model release |

## Nguồn dữ liệu

| Nguồn | Cỡ | Gate |
|---|---|---|
| `wikimedia/wikipedia` `20231101.vi` | ~1,5GB | không |
| `uonlp/CulturaX` subset `vi` | 55B token | **có** — đồng ý điều khoản trên trang dataset |
| `oscar-corpus/OSCAR-2301` vi | 68GB | có |
| NewsCorpus (binhvq) | 53GB | không |

Dự án chỉ dùng khoảng 6GB văn bản tiếng Việt cho lần train này.

## Giấy phép

[Apache License 2.0](LICENSE) — Copyright 2026 Lozens.

Lưu ý: giấy phép của repo/model không thay thế điều khoản hoặc giấy phép của các dataset nguồn.
