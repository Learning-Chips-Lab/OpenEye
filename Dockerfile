FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG VERILATOR_VERSION=5.036

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        build-essential \
        autoconf \
        flex \
        bison \
        help2man \
        perl \
        python3 \
        python3-venv \
        libfl-dev \
        zlib1g-dev \
        iverilog \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "v${VERILATOR_VERSION}" \
        https://github.com/verilator/verilator.git /tmp/verilator-src \
    && cd /tmp/verilator-src \
    && autoconf \
    && ./configure --prefix=/opt/verilator \
    && make -j4 \
    && make install \
    && rm -rf /tmp/verilator-src

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:/opt/verilator/bin:${PATH}"

RUN python -m pip install --no-cache-dir \
        "cocotb==2.1.0" \
        "numpy==2.5.4"
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpython3-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace

ENV PYTHONPATH="/workspace/src"
ENV SIM="icarus"

CMD ["bash", "test/cocotb_PE/run_recorded.sh", "pe_seed42"]