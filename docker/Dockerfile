FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace 

RUN apt-get update && \
    apt install -y \
    build-essential \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    ninja-build \
    libglib2.0-0 \
    libxrender-dev \
    libmagickwand-dev \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python
RUN apt-get update && \
    apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3-distutils python3-pip python3-apt

# Pytorch
# RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2

RUN pip install --upgrade pip

RUN pip install protobuf
# Detail

# gir clone https://github.com/zhanghang1989/detail-api.git /workspace/detail-api
# cd /workspace/detail-api/PythonAPI
# make 
# make install