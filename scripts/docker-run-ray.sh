#!/bin/bash
# Run Ray inference pipeline for GPU image classification
# Usage: ./scripts/docker-run-ray.sh [--input PATH] [--output PATH] [--gpu-workers N]

set -e

docker compose exec app bash -c "
    cd /app && \
    RAY_ADDRESS=ray://ray-head:10001 \
    python src/engine_comparison/distributed/ray_inference.py $@
"
