# Sử dụng Python slim để nhẹ hơn
FROM python:3.11-slim

# Cài các thư viện cần thiết
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 ffmpeg wget git curl && \
    rm -rf /var/lib/apt/lists/*

# Tạo thư mục app
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Cài dependencies (dùng --no-cache để tránh quá tải)
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code
COPY . .

# Mở port
EXPOSE 8000

# Chạy server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
