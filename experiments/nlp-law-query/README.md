# Legal Document Search API - Hướng dẫn triển khai

Hướng dẫn này sẽ giúp bạn triển khai và sử dụng Legal Document Search API.

## ⚙️ Cài đặt

### 1. (Tùy chọn) Tạo môi trường ảo

- **Trên Unix/macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **Trên Windows:**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

### 2. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
```

## Luồng chạy thông thường gồm 3 bước

Đầu vào là 5 file từ thư mục `data/input`.
Có tải stop words từ `src.stop_words`.

1.  **Preprocessing data** $\rightarrow$ dữ liệu dạng JSON (dữ liệu sau xử lý ở `data/processed`)
2.  **Embedding** $\rightarrow$ dữ liệu dạng vector (dữ liệu sau embedding ở `data/embeddings`). Bước này tốn nhiều thời gian để tạo embedding (5-10 phút với CPU). Nếu dữ liệu đã được xử lý và file embeddings đã tồn tại, có thể bỏ qua bước này hoặc tiếng hành chạy lại để processing các dữ liệu khác hoặc update embedding.
3.  **Search** $\rightarrow$ tìm kiếm dữ liệu (Cosine similarity).

### Chạy Preprocessing:

```bash
python -m src.data_preprocessing
```

hoặc

```bash
python src/data_preprocessing.py
```

hoặc

```bash
python3 src/data_preprocessing.py
```

### Chạy Embedding:

```bash
python -m src.embed
```

hoặc

```bash
python src/embed.py
```

hoặc

```bash
python3 src/embed.py
```

### Chạy Search:

```bash
python -m src.search_engine
```

hoặc

```bash
python src/search_engine.py
```

hoặc

```bash
python3 src/search_engine.py
```

## 🚀 Cách chạy dùng FastAPI

### 1. Chạy trực tiếp với Python

```bash
python main.py
```

API sẽ chạy tại: `http://localhost:8000`

### 2. Chạy với Docker

```bash
# Build Docker image
docker build -t backend .

# Chạy container
docker run -p 8000:8000 -v $(pwd)/data:/app/data backend
```

### 3. Chạy với Docker Compose (Khuyến nghị)

```bash
# Chạy với docker-compose
docker-compose up -d

# Xem logs
docker-compose logs -f

# Dừng
docker-compose down
```

## 🧪 Test API

### 1. Test tự động

```bash
# Chạy test script
python test_api.py
```

### 2. Test thủ công với curl

```bash
# Health check
curl http://localhost:8000/health

# Search
curl -X POST "http://localhost:8000/search" \
    -H "Content-Type: application/json" \
    -d '{"query": "Quy định về thời gian làm việc", "top_k": 5}'
```

### 3. Swagger UI

Truy cập: `http://localhost:8000/docs`