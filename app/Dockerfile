# Dùng image nhẹ, tối ưu cho CPU
FROM python:3.10-slim

# Cài các thư viện hệ thống cần thiết để chạy PyTorch + Pillow
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Tạo thư mục ứng dụng
WORKDIR /app

# Copy toàn bộ code vào container
COPY . /app

# Cài đặt thư viện Python
RUN pip install --no-cache-dir \
    Flask==3.1.0 \
    Flask-CORS==5.0.1 \
    onnxruntime==1.21.0 \
    torch==2.2.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    torchvision==0.17.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    Pillow \
    "numpy<2" \
    requests \
    datasets

# Port mặc định Flask
EXPOSE 5000

# Lệnh chạy app
CMD ["python", "app.py"]
