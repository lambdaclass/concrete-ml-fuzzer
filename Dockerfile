FROM --platform=linux/amd64 debian AS builder
WORKDIR /code
RUN apt update -y
RUN apt install -y build-essential \
                   libunwind-dev \
                   libssl-dev \
                   lldb \
                   pkg-config \
                   binutils-dev \
                   git \
                   python3 \
                   python3-pip
RUN pip install -U scikit-learn concrete-ml==0.5.1
