#!/bin/bash
# Upload sample datasets to MinIO for distributed pipeline testing
# Usage: ./scripts/upload-data.sh

set -e

echo "=== Uploading datasets to MinIO ==="

echo ""
echo "[1/3] Uploading Food-101 images to bucket/images..."
docker compose exec app python -m engine_comparison.cli.upload \
    --source .data/food101 \
    --dest bucket/images

echo ""
echo "[2/3] Creating image metadata parquet for Daft pipeline..."
docker compose exec app python -m engine_comparison.cli.create_image_metadata \
    --bucket bucket/images \
    --output bucket/image_metadata.parquet

echo ""
echo "[3/3] Uploading NYC Taxi data to lake/taxi..."
docker compose exec app python -m engine_comparison.cli.upload \
    --source .data/nyc_taxi \
    --dest lake/taxi

echo ""
echo "=== All uploads complete ==="
echo "  - Food-101 images: s3://bucket/images/"
echo "  - Image metadata:  s3://bucket/image_metadata.parquet"
echo "  - NYC Taxi:        s3://lake/taxi/"
