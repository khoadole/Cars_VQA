FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements từ thư mục app/
COPY app/requirements.txt* ./
RUN pip install --no-cache-dir \
    Flask==3.1.0 \
    Flask-CORS==5.0.1 \
    onnxruntime==1.21.0 \
    torch==2.2.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    torchvision==0.17.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    Pillow \
    "numpy<2" \
    requests \
    datasets \
    gunicorn

# Copy toàn bộ thư mục app/
COPY app/ /app/
RUN rm -f gunicorn.conf.py

# Tạo thư mục uploads
RUN mkdir -p uploads

ENV PORT=8080
EXPOSE $PORT

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120 --log-level info"]