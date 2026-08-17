# Trợ Lý Tài Liệu — RAG Chatbot đơn giản

Chatbot trả lời câu hỏi dựa trên tài liệu **PDF/DOCX** do bạn tải lên, dùng kiến
trúc **RAG (Retrieval-Augmented Generation)**: tài liệu được cắt nhỏ, embed,
lưu vào FAISS; khi hỏi, hệ thống tìm các đoạn liên quan nhất rồi đưa cho LLM
để sinh câu trả lời **kèm trích dẫn nguồn**.

Toàn bộ project bám sát 9 milestone (0 → 8) đã đề ra, mỗi milestone ứng với
một hoặc vài file trong `app/`.

```
PDF/DOCX → Text → Chunks → Embedding → FAISS → Retrieval → LLM → Answer + Citation
                                                                         ↑
                                                          FastAPI + Frontend (Milestone 7)
                                                          Docker/Log/Eval/Cache/Security (Milestone 8)
```

---

## Mục lục

- [Milestone 0 — Môi trường](#milestone-0--môi-trường)
- [Cài đặt & chạy nhanh](#cài-đặt--chạy-nhanh)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Milestone 1 — Document Processing](#milestone-1--document-processing)
- [Milestone 2 — Embedding](#milestone-2--embedding)
- [Milestone 3 — Vector Database](#milestone-3--vector-database)
- [Milestone 4 — Retrieval](#milestone-4--retrieval)
- [Milestone 5 — LLM](#milestone-5--llm)
- [Milestone 6 — RAG hoàn chỉnh](#milestone-6--rag-hoàn-chỉnh)
- [Milestone 7 — Web application](#milestone-7--web-application)
- [Milestone 8 — Production](#milestone-8--production)
- [Testing](#testing)
- [Giới hạn & hướng mở rộng](#giới-hạn--hướng-mở-rộng)

---

## Milestone 0 — Môi trường

Cài các công cụ nền tảng (làm một lần trên máy bạn):

1. **Python 3.10+** — kiểm tra bằng `python3 --version`. Nếu chưa có, tải tại
   [python.org/downloads](https://www.python.org/downloads/).
2. **Git** — kiểm tra bằng `git --version`. Tải tại [git-scm.com](https://git-scm.com/).
3. **VS Code** — tải tại [code.visualstudio.com](https://code.visualstudio.com/),
   cài thêm extension **Python** (Microsoft) và **Pylance**.
4. **GitHub** — tạo tài khoản, tạo repo trống, rồi từ thư mục project:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: RAG chatbot skeleton"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo>.git
   git push -u origin main
   ```
5. **Virtual environment** — tạo môi trường ảo riêng cho project để không
   xung đột package với các project khác (xem lệnh ở phần tiếp theo).

---

## Cài đặt & chạy nhanh

```bash
# 1) Tạo và kích hoạt virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) Cài dependencies (bản dev có thêm pytest/httpx để chạy test)
pip install -r requirements-dev.txt

# 3) Tạo file .env từ mẫu, rồi điền API key của bạn
cp .env.example .env
# Mở .env, điền OPENAI_API_KEY hoặc ANTHROPIC_API_KEY tuỳ theo LLM_PROVIDER
# Đổi luôn APP_API_KEY thành một chuỗi bí mật của riêng bạn

# 4) Chạy server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# (hoặc: make run)
```

Mở trình duyệt tại **http://localhost:8000** → dán giá trị `APP_API_KEY` vào
ô "X-API-Key" ở góc trên → tải lên một file PDF/DOCX → đặt câu hỏi.

> Lần chạy đầu tiên sẽ tự tải model embedding (~90 MB) từ Hugging Face, cần
> có Internet. Từ lần sau model được cache lại, không cần mạng cho bước này.

---

## Cấu trúc thư mục

```
rag-chatbot/
├── app/
│   ├── config.py             # Milestone 0 — cấu hình tập trung, đọc từ .env
│   ├── logger.py             # Milestone 8 — logging ra console + file
│   ├── document_processor.py # Milestone 1 — PDF/DOCX -> text -> chunks
│   ├── embeddings.py         # Milestone 2 — text -> vector
│   ├── vector_store.py       # Milestone 3 — FAISS index + metadata
│   ├── retrieval.py          # Milestone 4 — question -> top-K chunks
│   ├── llm.py                 # Milestone 5 — context + question -> answer
│   ├── cache.py               # Milestone 8 — cache câu trả lời
│   ├── rag_pipeline.py        # Milestone 6 — nối tất cả lại + citation
│   └── main.py                 # Milestone 7+8 — FastAPI, auth, rate limit
├── static/index.html          # Milestone 7 — giao diện chat (1 file HTML)
├── scripts/evaluate.py        # Milestone 8 — script đánh giá chất lượng
├── eval/questions.json        # bộ câu hỏi mẫu cho evaluate.py
├── tests/                     # unit test + API test (pytest)
├── Dockerfile, docker-compose.yml, Makefile   # Milestone 8
├── requirements.txt / requirements-dev.txt
└── .env.example
```

---

## Milestone 1 — Document Processing

`app/document_processor.py`

- `extract_pages_from_pdf()` dùng **pypdf**, giữ lại số trang để trích dẫn sau này.
- `extract_text_from_docx()` dùng **python-docx**, đọc cả đoạn văn lẫn bảng.
- `chunk_text()` là bộ chia đoạn tự viết (không cần LangChain): ưu tiên cắt
  theo ranh giới đoạn văn, chỉ cắt theo câu khi một đoạn quá dài, luôn giữ
  phần **overlap** giữa các chunk để không mất ngữ cảnh ở điểm cắt.
- Tham số `CHUNK_SIZE` / `CHUNK_OVERLAP` cấu hình trong `.env`.

## Milestone 2 — Embedding

`app/embeddings.py`

- Dùng **sentence-transformers**, mặc định model
  `all-MiniLM-L6-v2` (384 chiều, nhẹ, chạy tốt trên CPU, chạy hoàn toàn local
  — không tốn phí, không cần API key).
- Model được load một lần (`lru_cache`) và tái sử dụng cho cả việc embed tài
  liệu lẫn embed câu hỏi, đảm bảo cùng không gian vector.
- Vector được chuẩn hoá (`normalize_embeddings=True`) để tích vô hướng
  (inner product) tương đương cosine similarity — khớp với `IndexFlatIP`
  của FAISS ở Milestone 3.

## Milestone 3 — Vector Database

`app/vector_store.py`

- `FaissVectorStore` bọc một `faiss.IndexFlatIP` cùng danh sách metadata
  (`Chunk`) song song, để map kết quả tìm kiếm về đúng văn bản gốc.
- `save()`/`load()` lưu cả index lẫn metadata xuống đĩa
  (`VECTOR_STORE_DIR`), nên dữ liệu vẫn còn sau khi restart server.
- `exists()` cho phép kiểm tra store đã có trên đĩa hay chưa **mà không cần
  load model embedding** — giúp `/health` phản hồi nhanh và nhẹ.

## Milestone 4 — Retrieval

`app/retrieval.py`

- `retrieve(question, store, top_k)`: embed câu hỏi bằng đúng model ở
  Milestone 2, tìm `top_k` chunk gần nhất trong FAISS, trả về kèm điểm
  tương đồng (cosine similarity).

## Milestone 5 — LLM

`app/llm.py`

- Hỗ trợ **cả OpenAI lẫn Anthropic**, chọn qua `LLM_PROVIDER` trong `.env`
  — chỉ cần cấu hình API key của nhà cung cấp bạn dùng.
- Prompt hệ thống yêu cầu mô hình **chỉ trả lời dựa trên context được cung
  cấp**, nói rõ khi không đủ thông tin thay vì bịa, và chú thích nguồn theo
  dạng `[1]`, `[2]`.
- Có retry (thư viện `tenacity`) để tự thử lại khi gặp lỗi mạng/API tạm thời.

## Milestone 6 — RAG hoàn chỉnh

`app/rag_pipeline.py`

- `RAGPipeline.ingest(file)`: Milestone 1 → 2 → 3 (xử lý file, embed, lưu
  vào FAISS).
- `RAGPipeline.ask(question)`: Milestone 4 → 5 (retrieval → LLM), rồi đóng
  gói kết quả thành `RAGAnswer(answer, citations)`. Citation được lấy
  **trực tiếp từ metadata đã retrieve** (nguồn, trang, điểm số, đoạn trích)
  — không phụ thuộc hoàn toàn vào việc LLM tự trích dẫn đúng, nên trích dẫn
  luôn đáng tin cậy.

## Milestone 7 — Web application

`app/main.py` + `static/index.html`

FastAPI phục vụ cả API lẫn giao diện tĩnh:

| Method | Endpoint         | Mô tả                                             |
|--------|------------------|----------------------------------------------------|
| GET    | `/health`        | Kiểm tra server còn sống (không cần API key)        |
| POST   | `/api/upload`    | Tải PDF/DOCX, xử lý và thêm vào vector store         |
| POST   | `/api/ask`       | Gửi câu hỏi, nhận `{answer, citations}`             |
| GET    | `/api/documents` | Danh sách tài liệu đã nạp + tổng số chunk            |
| GET    | `/`              | Giao diện chat (static/index.html)                  |

Các endpoint `/api/*` yêu cầu header `X-API-Key` khớp với `APP_API_KEY`
trong `.env` (xem Milestone 8).

Ví dụ gọi bằng `curl`:
```bash
curl -X POST http://localhost:8000/api/upload \
  -H "X-API-Key: <APP_API_KEY của bạn>" \
  -F "file=@/duong/dan/tai-lieu.pdf"

curl -X POST http://localhost:8000/api/ask \
  -H "X-API-Key: <APP_API_KEY của bạn>" \
  -H "Content-Type: application/json" \
  -d '{"question": "Tài liệu này nói về gì?"}'
```

## Milestone 8 — Production

**Docker**
```bash
docker compose up -d --build     # hoặc: make docker-up
```
`docker-compose.yml` mount thư mục `data/` và `logs/` ra ngoài container để
dữ liệu (vector store, cache, log) không mất khi container bị xoá.

**Logging** (`app/logger.py`) — ghi log ra console **và** file xoay vòng
(`logs/app.log`, tối đa 5 MB × 3 file) cho mọi bước xử lý tài liệu, truy
vấn, và lỗi.

**Evaluation** (`scripts/evaluate.py`) — chạy một bộ câu hỏi mẫu
(`eval/questions.json`) qua toàn bộ pipeline, đo:
- **Retrieval hit rate**: tài liệu mong đợi có nằm trong citation không.
- **Keyword coverage**: câu trả lời có chứa các từ khoá mong đợi không.
- **Latency** trung bình mỗi câu hỏi.
Kết quả chi tiết từng câu ghi ra `eval/report.csv`. Chỉnh `eval/questions.json`
theo tài liệu thật của bạn rồi chạy `make evaluate`.

**Caching** (`app/cache.py`) — cache câu trả lời theo nội dung câu hỏi
(dùng `diskcache`, có TTL qua `CACHE_TTL_SECONDS`), tránh gọi lại LLM (tốn
tiền + chậm) cho câu hỏi trùng lặp.

**Security** (`app/main.py`)
- Xác thực bằng shared-secret qua header `X-API-Key` (`APP_API_KEY`).
- Rate limiting theo API key, mặc định `RATE_LIMIT_PER_MINUTE=20` request/phút
  — cài đặt in-memory, phù hợp một instance; nếu chạy nhiều worker/replica,
  hãy thay bằng giải pháp dùng Redis (vd. `slowapi` + Redis) để các worker
  dùng chung bộ đếm.
- Giới hạn loại file (`.pdf`/`.docx`) và dung lượng (`MAX_UPLOAD_MB`).
- Tên file được làm sạch (bỏ path, thêm prefix ngẫu nhiên) trước khi lưu,
  tránh path traversal / ghi đè file.
- `.env` (chứa secret) nằm trong `.gitignore`, không bao giờ commit.

**Deployment** — vài lựa chọn cho một tool "đơn giản" như thế này:
- VPS bất kỳ (Ubuntu) + Docker + reverse proxy (Caddy/Nginx) để có HTTPS
  miễn phí qua Let's Encrypt.
- Nền tảng PaaS (Render, Railway, Fly.io...): deploy trực tiếp từ Dockerfile.
- Luôn nhớ set các biến môi trường trong `.env.example` trên nền tảng deploy,
  và thu hẹp `allow_origins` trong `CORSMiddleware` (`app/main.py`) về đúng
  domain frontend thay vì `"*"` khi lên production thật.

---

## Testing

```bash
pytest -v          # hoặc: make test
```

`tests/test_basic.py` kiểm tra chunking và FAISS store bằng vector giả lập
(không cần tải model). `tests/test_api.py` dùng `TestClient` của FastAPI để
kiểm tra auth, rate limit, và validate upload — cũng không cần mạng, nên
chạy được cả trong môi trường CI không có Internet.

---

## Giới hạn & hướng mở rộng

- **PDF scan/ảnh** chưa hỗ trợ OCR — `pypdf` chỉ đọc được text layer có sẵn.
  Muốn hỗ trợ PDF scan, thêm `pytesseract` + `pdf2image` vào bước extract.
- Rate limiter hiện tại là in-memory, **không** đồng bộ giữa nhiều worker —
  đủ dùng cho một instance nhỏ, cần Redis nếu scale ngang.
- Chunking hiện dựa trên số ký tự + ranh giới câu/đoạn đơn giản; có thể
  thay bằng `RecursiveCharacterTextSplitter` của LangChain nếu cần tinh
  chỉnh sâu hơn.
- Chưa có xoá/quản lý từng tài liệu riêng lẻ trong vector store (chỉ có
  thêm) — có thể mở rộng bằng cách lưu thêm `doc_id` và lọc khi xoá.
- Đánh giá (Milestone 8) hiện là rule-based (khớp từ khoá/nguồn); nếu cần
  đánh giá ngữ nghĩa sâu hơn, cân nhắc tích hợp RAGAS hoặc chấm điểm bằng
  chính LLM (LLM-as-judge).
