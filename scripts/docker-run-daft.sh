#!/bin/bash
# Run Daft document embedding pipeline (uses Ray as backend)
# Usage: ./scripts/docker-run-daft.sh [--input PATH] [--output PATH]

set -e

docker compose exec app bash -c "
    cd /app && \
    DAFT_RUNNER=ray \
    RAY_ADDRESS=ray://ray-head:10001 \
    python src/engine_comparison/distributed/daft_pipeline.py $@
"
