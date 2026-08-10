# Sử dụng Base Image PyTorch 2.3.0 hỗ trợ CUDA 12.1 & Python 3.10
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Cấu hình biến môi trường không ghi .pyc và xuất log trực tiếp
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các gói hệ thống cần thiết (FFmpeg, OpenCV, Git, Build Tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements.txt và cài đặt phụ thuộc Python
COPY requirements.txt .
RUN grep -v '^chumpy' requirements.txt > requirements-no-chumpy.txt && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-no-chumpy.txt

# Install chumpy separately
RUN python -m pip install --no-cache-dir --no-build-isolation chumpy==0.70

# Copy mã nguồn ứng dụng vào container
COPY . .

# Tạo cấu trúc thư mục I/O cần thiết
RUN mkdir -p input/temp_upload output/result inputs/checkpoints

# Mở port 8000 cho FastAPI
EXPOSE 8000

# Khởi chạy Uvicorn Server
CMD ["uvicorn", "gvhmr-api:app", "--host", "0.0.0.0", "--port", "8000"]
