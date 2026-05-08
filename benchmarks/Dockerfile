# syntax=docker/dockerfile:1.7

FROM nvidia/cuda:13.1.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/go/bin:${PATH}"

ARG MOONCAKE_REPO=https://github.com/Dao007forever/Mooncake.git
ARG MOONCAKE_REF=main
ARG CMAKE_BUILD_TYPE=Release

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        curl \
        wget \
        unzip \
        build-essential \
        cmake \
        ninja-build \
        pkg-config \
        patchelf \
        python3 \
        python3-dev \
        python3-pip \
        python-is-python3 \
        libibverbs-dev \
        rdma-core \
        ibverbs-providers \
        librdmacm-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libjsoncpp-dev \
        libunwind-dev \
        libnuma-dev \
        libpython3-dev \
        libboost-all-dev \
        libssl-dev \
        libgrpc-dev \
        libgrpc++-dev \
        libprotobuf-dev \
        libyaml-cpp-dev \
        protobuf-compiler-grpc \
        libcurl4-openssl-dev \
        libhiredis-dev \
        liburing-dev \
        libjemalloc-dev \
        libmsgpack-dev \
        libzstd-dev \
        libasio-dev \
        libxxhash-dev \
        libc6-dev \
        libc-bin && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone --branch "${MOONCAKE_REF}" --recursive "${MOONCAKE_REPO}" Mooncake

WORKDIR /workspace/Mooncake

RUN bash dependencies.sh -y

RUN cmake -S . -B build -G Ninja \
        -DBUILD_UNIT_TESTS=OFF \
        -DUSE_HTTP=ON \
        -DUSE_ETCD=ON \
        -DUSE_CUDA=ON \
        -DWITH_STORE=ON \
        -DSTORE_USE_ETCD=ON \
        -DPython3_EXECUTABLE=/usr/bin/python3 \
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" && \
    LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH:-}" \
    cmake --build build -j"$(nproc)"

CMD ["/bin/bash"]
