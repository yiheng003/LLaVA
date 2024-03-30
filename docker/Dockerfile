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
    apt-get install -y python3.9 python3.9-venv python3-distutils python3-pip python3-apt

# update alternatives to make python3.9 the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# manage package in virtual environment 
RUN mkdir /workspace/environment
RUN python3 -m venv /workspace/environment
RUN echo 'source /workspace/environment/bin/activate' >> /root/.bashrc

RUN pip install --upgrade pip