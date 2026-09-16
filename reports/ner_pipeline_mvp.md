# MVP Pipeline: Trích xuất NER từ Municipal Chatbot Dataset — tối ưu cho GLM-5.3-Flash

**Input:** `Engdata(usage_binatanya_01-03-25_16-11-).csv` — cột `user_message_en`, `agent_response_en`
**Inference engine:** GLM-5.3-Flash (Z.ai) — MoE 320B total / 18B active, OpenAI-compatible API, context 1M token, hỗ trợ function calling + reasoning-effort control (`low`/`high`/`max`), giá tham khảo ~$0.15/1M input, $0.5/1M output, cache read $0.03/1M (giá theo provider, cần verify lại tại thời điểm chạy vì có ~18 provider host model này với giá/tốc độ khác nhau — Zai chính chủ, Novita, Together, DeepInfra, Fireworks...).

Bối cảnh: paper hướng **no-labelling / descriptive audit**. Dùng LLM zero-shot làm NER extractor thay vì fine-tune vẫn giữ được tinh thần "không gán nhãn tay để train" — nhưng cần lưu ý khi viết Methods: đây là *prompt-based zero-shot extraction*, khác về bản chất reproducibility so với pretrained spaCy (xem mục 6).

---

## 1. Vì sao đổi từ spaCy sang GLM-5.3-Flash

| | spaCy `en_core_web_trf` | GLM-5.3-Flash (LLM zero-shot) |
|---|---|---|
| Entity set | Cố định 18 loại OntoNotes (generic) | **Tự định nghĩa taxonomy theo domain municipal** — ưu điểm lớn nhất |
| Domain fit | Không biết tên phường/phòng ban địa phương | Hiểu ngữ cảnh, suy luận được entity đặc thù (VD "giấy phép xây dựng", "hoá đơn nước") |
| Setup | Cần GPU để chạy nhanh trên corpus lớn | Gọi API, không cần hạ tầng GPU riêng |
| Cost | Free (local) | Trả theo token — cần ước tính trước |
| Reproducibility | Deterministic 100% | Cần cố định temperature=0, model version, prompt để tối đa hoá reproducibility |
| Rủi ro | Miss domain-specific entity | **Hallucination**: có thể "bịa" entity không có trong text gốc — bắt buộc phải verify |

**Quyết định MVP:** GLM-5.3-Flash làm engine chính, với taxonomy tự định nghĩa; spaCy giữ lại làm **cross-check nhỏ trên sample** để đo agreement (thay vì ngược lại như bản trước).

## 2. Taxonomy entity tuỳ chỉnh cho domain municipal

Thay vì generic PERSON/ORG/GPE, định nghĩa entity type bám sát nội dung chatbot. Taxonomy này xây dựng độc lập, KHÔNG dựa vào `sector_labels.csv`/`topic_keywords.csv` đã có — vì đó là output của bước unsupervised khác (LDA/rule-based proxy) chưa được xác thực, dùng nó để định hình taxonomy sẽ khiến sai số của bước đó lan sang NER (circular, khó biện minh trong Methods):

```
LOCATION        — địa danh, khu vực, đường, phường
DEPARTMENT      — phòng ban/cơ quan municipal được nhắc tới
DATE_TIME       — ngày giờ, hạn chót, lịch hẹn
MONEY_FEE       — số tiền, phí, lệ phí
PERMIT_DOC      — loại giấy phép/hồ sơ/biểu mẫu
SERVICE_REQUEST — loại dịch vụ/yêu cầu công dân đề cập (rác, nước, xây dựng...)
CONTACT_INFO    — số điện thoại/email (để tách riêng, tránh lẫn vào CARDINAL)
LAW_REGULATION  — luật/quy định/nghị quyết được viện dẫn
PERSON          — tên người (nếu có — cần cẩn trọng privacy, xem mục 7)
```

Taxonomy này là *hypothesis ban đầu*, xây dựng từ đọc trực tiếp một số message mẫu (không qua trung gian topic/sector file) — nên chạy thử trên ~50 sample, xem GLM trả về gì ngoài danh sách trên (entity nào bị bỏ sót, entity nào không khớp label nào), rồi tinh chỉnh lại 1 vòng trước khi chạy full.

## 3. Structured output — dùng function calling, không parse free text

GLM-5.3-Flash OpenAI-compatible nên định nghĩa 1 tool/function `extract_entities` với JSON schema cố định, ép model trả đúng cấu trúc thay vì parse text tự do (giảm lỗi parse, tăng tính reproducible):

```json
{
  "name": "extract_entities",
  "description": "Extract named entities from a municipal chatbot message",
  "parameters": {
    "type": "object",
    "properties": {
      "entities": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "text": {"type": "string", "description": "exact substring copied verbatim from input"},
            "label": {"type": "string", "enum": ["LOCATION","DEPARTMENT","DATE_TIME","MONEY_FEE",
                                                   "PERMIT_DOC","SERVICE_REQUEST","CONTACT_INFO",
                                                   "LAW_REGULATION","PERSON"]}
          },
          "required": ["text", "label"]
        }
      }
    },
    "required": ["entities"]
  }
}
```

System prompt cốt lõi (giữ ngắn, cố định để reproducible):
> "You are a precise information-extraction system. Extract entities strictly using the provided taxonomy. Copy `text` VERBATIM from the input — do not paraphrase, translate, or normalize. If no entity of a type exists, omit it. Do not invent entities."

## 4. Batching & cost/latency optimization

- **Reasoning effort = `low`**: NER là extraction task, không cần deep reasoning → set thấp nhất để giảm token sinh ra ở phần "thinking" (model không tắt được thinking hoàn toàn nhưng effort thấp giảm đáng kể chi phí/latency).
- **Temperature = 0** (hoặc gần 0) để tối đa determinism.
- **Batch nhiều message/1 call**: tận dụng context 1M token — gộp ví dụ 20–30 message ngắn/call, đánh số `msg_id` trong prompt, yêu cầu trả về array kết quả theo `msg_id` tương ứng. Giảm số round-trip API → giảm overhead latency + phí input lặp lại system prompt.
  - Trade-off: batch càng lớn, 1 lỗi JSON làm hỏng cả batch → cần retry granular (nếu batch fail, chia đôi rồi thử lại — binary-split retry) thay vì fail cả batch to.
- **Concurrency có kiểm soát**: async request với semaphore giới hạn (VD 5–10 request đồng thời), backoff khi rate-limit (429).
- **Ước tính cost trước khi chạy full**: `total_cost ≈ (n_messages/batch_size) × (avg_input_tokens_per_batch × $0.15/1M + avg_output_tokens_per_batch × $0.5/1M)`. Chạy thử trên `--sample 200` trước, đo token usage thực tế từ response (`usage.prompt_tokens`, `usage.completion_tokens`), rồi extrapolate ra full corpus — tránh chạy nền cả đêm rồi mới biết cost.

## 5. Verification bắt buộc — chống hallucination

Vì LLM có thể "bịa" entity, **mọi entity trả về phải được kiểm tra là substring (hoặc fuzzy match ≥95%) của message gốc** trước khi ghi vào `entities.csv`. Entity không match → log riêng ra `unverified_entities.csv` để review, KHÔNG tính vào thống kê chính thức của paper. Đây cũng là 1 chỉ số hay để báo cáo: "% hallucination rate" của model trên corpus này.

## 6. Kiến trúc pipeline (cập nhật)

```
CSV raw
  │
  ▼
[1] Ingest & Clean            ──▶ clean_messages.parquet
  │
  ▼
[2] Batch & Prompt Build      ──▶ batches (msg_id → text)
  │
  ▼
[3] GLM-5.3-Flash Inference   ──▶ entities_raw.csv (+ usage log token/cost)
      (function calling, temp=0, reasoning=low, async + retry)
  │
  ▼
[4] Verbatim Verification     ──▶ entities.csv (verified) + unverified_entities.csv
  │
  ▼
[5] Aggregate                 ──▶ entity_freq.csv, entity_by_conv.csv
  │
  ▼
[6] Cross-analysis            ──▶ entity_by_sector.csv, entity_by_topic.csv
  │
  ▼
[7] Cross-check sample vs spaCy en_core_web_trf ──▶ agreement_report.md
```

## 7. Privacy / data-handling — cần xử lý TRƯỚC khi gửi API ngoài

Đây là điểm khác biệt quan trọng nhất so với chạy spaCy local: **dữ liệu message của công dân thật sẽ được gửi ra ngoài tới API provider thứ ba** (Zai/Novita/Together/...). Cần làm rõ trước khi chạy full corpus:

1. Kiểm tra data-sharing agreement với phía cung cấp dataset (municipality) — có cho phép gửi dữ liệu (dù đã dịch, ẩn danh phần nào) ra API bên thứ ba không.
2. Chạy 1 lớp **PII scrub bằng regex** trước khi gửi: số điện thoại, email, số CMND/ID nếu có, trước khi đưa vào prompt — kể cả khi bạn cũng muốn trích xuất `CONTACT_INFO`/`PERSON` làm entity, có thể mask giá trị thật trong prompt gửi đi và chỉ giữ vị trí/label, hoặc log riêng phần PII ở máy local.
3. Ưu tiên chọn provider chính chủ (Zai official platform) hoặc provider có chính sách không lưu/không train trên dữ liệu gửi lên (data retention policy) — kiểm tra ToS trước khi chạy corpus thật.
4. Vì GLM-5.3-Flash là open-weight (có trên Hugging Face), nếu dataset quá nhạy cảm để gửi ra ngoài, phương án dự phòng là **self-host** (vLLM/SGLang) — nhưng 320B total params vẫn cần hạ tầng GPU đáng kể dù chỉ 18B active/token (MoE vẫn phải load toàn bộ weight vào VRAM); cần cân nhắc chi phí hạ tầng vs. dùng API.

## 8. Script skeleton

```python
# ner_glm_pipeline.py
import argparse, asyncio, json, logging, os, time
import pandas as pd
from openai import AsyncOpenAI
from rapidfuzz import fuzz

TAXONOMY = ["LOCATION","DEPARTMENT","DATE_TIME","MONEY_FEE","PERMIT_DOC",
            "SERVICE_REQUEST","CONTACT_INFO","LAW_REGULATION","PERSON"]

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract named entities per message from a batch",
        "parameters": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "msg_id": {"type": "string"},
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string"},
                                        "label": {"type": "string", "enum": TAXONOMY}
                                    },
                                    "required": ["text", "label"]
                                }
                            }
                        },
                        "required": ["msg_id", "entities"]
                    }
                }
            },
            "required": ["results"]
        }
    }
}

SYSTEM_PROMPT = (
    "You are a precise information-extraction system. Extract entities strictly "
    "using the provided taxonomy. Copy `text` VERBATIM from the input — do not "
    "paraphrase, translate, or normalize. If no entity of a type exists, omit it. "
    "Do not invent entities."
)

def build_batch_prompt(batch: list[tuple[str, str]]) -> str:
    # batch: list of (msg_id, text)
    lines = [f"[{mid}] {text}" for mid, text in batch]
    return "Extract entities for each numbered message:\n\n" + "\n".join(lines)

async def call_glm(client: AsyncOpenAI, batch, model, semaphore, retries=3):
    prompt = build_batch_prompt(batch)
    async with semaphore:
        for attempt in range(retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    reasoning_effort="low",
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": prompt}],
                    tools=[EXTRACT_TOOL],
                    tool_choice={"type": "function", "function": {"name": "extract_entities"}},
                )
                call = resp.choices[0].message.tool_calls[0]
                parsed = json.loads(call.function.arguments)
                usage = resp.usage
                return parsed["results"], usage
            except Exception as e:
                logging.warning(f"batch failed (attempt {attempt}): {e}")
                if len(batch) > 1 and attempt == retries - 1:
                    # binary-split retry thay vì fail cả batch
                    mid = len(batch) // 2
                    left, _ = await call_glm(client, batch[:mid], model, semaphore, retries)
                    right, _ = await call_glm(client, batch[mid:], model, semaphore, retries)
                    return left + right, None
                await asyncio.sleep(2 ** attempt)
    return [], None

def verify_verbatim(text_original: str, entity_text: str, threshold=95) -> bool:
    if entity_text in text_original:
        return True
    return fuzz.partial_ratio(entity_text, text_original) >= threshold

# orchestrate: load -> clean -> chunk into batches -> asyncio.gather(call_glm...)
# -> verify_verbatim mỗi entity -> ghi entities.csv / unverified_entities.csv
# -> log usage.prompt_tokens/completion_tokens mỗi batch ra cost_log.csv
# -> checkpoint resume theo msg_id cuối đã xử lý (giống ner.pid/ner_full.log hiện có)
```

**Trước khi chạy full corpus:** luôn `--sample 200` trước để (a) tinh chỉnh taxonomy, (b) đo hallucination rate, (c) đo cost thực tế/1000 message rồi extrapolate.

## 9. Validation & agreement check

- Chạy song song trên cùng sample: GLM-5.3-Flash (reasoning=low) vs GLM-5.3-Flash (reasoning=high) — agreement rate cao thì low đã đủ dùng cho full corpus (tiết kiệm cost).
- Chạy spaCy `en_core_web_trf` trên cùng sample, so % overlap cho các label tương đương (LOCATION↔GPE/LOC, DATE_TIME↔DATE/TIME, PERSON↔PERSON) — đưa vào Methods như 1 inter-method agreement metric, tương tự cách paper đã làm validation cho LDA.
- Spot-check thủ công 50 message (face validity), không dùng để train.

## 10. Rủi ro & giới hạn cần nêu trong Methods

- Zero-shot LLM extraction có thể hallucinate — đã có bước verify verbatim (mục 5) nhưng không loại bỏ hoàn toàn rủi ro nhầm label đúng nhưng text sai ngữ cảnh.
- Taxonomy tự định nghĩa mang tính chủ quan của nhóm nghiên cứu — cần nêu rõ đây là hypothesis-driven categories, không phải chuẩn ngành.
- Gửi dữ liệu ra API bên thứ ba — cần nêu rõ trong phần Ethics/Data Availability của paper là dữ liệu đã qua xử lý gì trước khi gửi (anonymization).
- Kết quả phụ thuộc vào model snapshot tại thời điểm chạy (Zai có thể update model) — nên log rõ ngày chạy, model version string trả về trong response, provider cụ thể đã dùng.

## 11. Việc tiếp theo sau MVP
- Nếu hallucination rate thấp và agreement với spaCy cao trên sample, mở rộng full corpus.
- Cân nhắc thêm 1 pass riêng để resolve/canonical hoá entity (VD gom biến thể tên địa danh) giống bản trước.
- Đưa taxonomy cuối cùng + prompt cố định vào Appendix của paper để đảm bảo reproducibility.
