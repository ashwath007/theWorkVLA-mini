# ---- Stage 1: Base system dependencies ----
FROM python:3.11-slim AS base

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libopencv-dev \
    python3-opencv \
    libsndfile1 \
    portaudio19-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    libtbb2 \
    libtbb-dev \
    libdc1394-dev \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Stage 2: Python dependencies ----
FROM base AS builder

WORKDIR /build

# Copy and install Python requirements
COPY requirements.txt .

# Install Python dependencies into a separate location for layer caching
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ---- Stage 3: Runtime image ----
FROM base AS runtime

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Create data directories
RUN mkdir -p /data/sessions /models /app/logs /app/checkpoints

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY setup.py .
COPY requirements.txt .

# Install the package in development mode
RUN pip install -e . --no-deps

# Set environment variables
ENV DATA_DIR=/data/sessions
ENV MODEL_OUTPUT_DIR=/models
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose API port
EXPOSE 8000

# Default: run API server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
