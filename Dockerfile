# syntax=docker/dockerfile:1

FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies required by OpenCV and others
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
       ca-certificates \
       ffmpeg \
       libsm6 \
       libxext6 \
       libglib2.0-0 \
       libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application code
COPY ML_api ./ML_api

# Ensure model is available at the expected path per Config.ONNX_MODEL_PATH
# Config expects: ML_api/model/rice_disease_model.onnx
RUN mkdir -p ML_api/model \
    && if [ -f ML_api/app/model/rice_disease_model.onnx ]; then \
         cp ML_api/app/model/rice_disease_model.onnx ML_api/model/; \
       fi

WORKDIR /app/ML_api

# App configuration
ENV HOST=0.0.0.0 \
    PORT=8001 \
    DEBUG=false \
    USE_ONNX=true

EXPOSE 8001

# Default command
CMD ["python", "run.py"]


