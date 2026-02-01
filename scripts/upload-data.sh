#!/bin/bash
# Upload local data to MinIO for distributed pipeline testing
# Usage: ./scripts/upload-data.sh [local_path] [bucket/path]
#
# Example:
#   ./scripts/upload-data.sh .data/food101 bucket/images/

set -e

LOCAL_PATH="${1:-.data}"
REMOTE_PATH="${2:-lake/data}"

echo "Uploading $LOCAL_PATH to s3://$REMOTE_PATH ..."

docker compose exec minio mc alias set local http://localhost:9000 minioadmin minioadmin 2>/dev/null || true
docker compose exec minio mc cp --recursive "/data/$LOCAL_PATH" "local/$REMOTE_PATH"

echo "✓ Upload complete: s3://$REMOTE_PATH"
