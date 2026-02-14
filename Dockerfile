# Multi-stage build for nordocr server.
# Requires NVIDIA Container Toolkit for GPU access.

# Stage 1: Build
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

# Install Rust toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev clang \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install TensorRT (adjust version as needed).
# In production: use the NVIDIA TensorRT container or install from .deb.
# RUN apt-get install -y libnvinfer-dev libnvinfer-plugin-dev

WORKDIR /app
COPY . .

# Build release binary.
RUN cargo build --release --bin nordocr

# Stage 2: Runtime
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Install TensorRT runtime libraries.
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libnvinfer10 libnvinfer-plugin10 \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary and model files.
COPY --from=builder /app/target/release/nordocr /app/nordocr
COPY models/ /app/models/

# Health check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

ENTRYPOINT ["/app/nordocr"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
