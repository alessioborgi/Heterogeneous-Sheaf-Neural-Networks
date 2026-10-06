# CUDA-ready image for HetSheaf (Linux / CUDA 12.8 wheels, see pyproject.toml).
#
# Build:
#   docker build -t hetsheaf:latest .
# Run (mount the repo and a data directory, expose all GPUs):
#   docker run -it --rm --gpus all --ipc host \
#     -v "$PWD":/work/project -v /path/to/data:/work/project/data \
#     hetsheaf:latest /bin/bash
# podman users: replace `--gpus all` with `--device nvidia.com/gpu=all`.

FROM python:3.13-slim

RUN apt update && apt install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    swig \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel uv
RUN mkdir /.cache && chmod 777 /.cache

WORKDIR /work/project
COPY pyproject.toml uv.lock README.md ./
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv sync --frozen --no-install-project
ENV PATH="/opt/venv/bin:$PATH"
