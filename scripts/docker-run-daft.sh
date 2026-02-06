#!/bin/bash
# Run Daft image embedding pipeline (uses Ray as backend)
# Usage: ./scripts/docker-run-daft.sh [--input PATH] [--output PATH]

set -e

docker compose exec -e DAFT_RUNNER=ray -e RAY_ADDRESS=ray://ray-head:10001 app \
    python src/engine_comparison/distributed/daft_pipeline.py "$@"

