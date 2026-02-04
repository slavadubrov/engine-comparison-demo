#!/bin/bash
# Run Ray inference pipeline for GPU image classification
# Usage: ./scripts/docker-run-ray.sh [--input PATH] [--output PATH] [--gpu-workers N]

set -e

docker compose exec -e RAY_ADDRESS=auto \
    -e AWS_ENDPOINT_URL=http://minio:9000 \
    -e AWS_ACCESS_KEY_ID=minioadmin \
    -e AWS_SECRET_ACCESS_KEY=minioadmin \
    ray-head python /app/src/engine_comparison/distributed/ray_inference.py "$@"

