#!/usr/bin/env python3
"""
Create image metadata parquet for Daft pipeline.

Scans uploaded images in S3 and creates a parquet file with their URLs.

Usage:
    python -m engine_comparison.cli.create_image_metadata --bucket bucket/images --output bucket/image_metadata.parquet
"""

from __future__ import annotations

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
import s3fs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create image metadata parquet from S3 images",
    )
    parser.add_argument(
        "--bucket",
        "-b",
        default="bucket/images",
        help="S3 bucket/prefix containing images (e.g., 'bucket/images')",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="bucket/image_metadata.parquet",
        help="Output path for metadata parquet",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("AWS_ENDPOINT_URL", "http://minio:9000"),
        help="S3 endpoint URL",
    )

    args = parser.parse_args()

    # Connect to S3
    fs = s3fs.S3FileSystem(
        endpoint_url=args.endpoint,
        key=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        secret=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
    )

    # List all images
    image_files = fs.ls(args.bucket)
    image_urls = [
        f"s3://{f}" for f in image_files if f.endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(image_urls)} images in s3://{args.bucket}")

    # Create and write parquet
    table = pa.table({"image_url": image_urls})

    with fs.open(args.output, "wb") as f:
        pq.write_table(table, f)

    print(f"✓ Created s3://{args.output}")


if __name__ == "__main__":
    main()
