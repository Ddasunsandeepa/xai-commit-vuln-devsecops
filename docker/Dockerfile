FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip

# CUDA-enabled PyTorch
RUN python -m pip install torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cu126

# Remaining research dependencies
RUN python -m pip install -r requirements.txt

CMD ["bash"]
