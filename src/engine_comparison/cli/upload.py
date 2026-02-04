#!/usr/bin/env python3
"""
Data Upload CLI - Upload local data to MinIO/S3 for distributed pipelines.

Usage:
    python -m engine_comparison.cli.upload --source .data/food101 --dest bucket/images/

Or via docker compose:
    docker compose exec app python -m engine_comparison.cli.upload --source .data/food101 --dest bucket/images/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import s3fs


def get_s3_client(
    endpoint_url: str = "http://minio:9000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
) -> s3fs.S3FileSystem:
    """Create S3 filesystem client configured for MinIO."""
    return s3fs.S3FileSystem(
        endpoint_url=endpoint_url,
        key=access_key,
        secret=secret_key,
    )


def upload_path(
    fs: s3fs.S3FileSystem,
    source: Path,
    dest: str,
    verbose: bool = True,
) -> int:
    """
    Upload a file or directory to S3.

    Args:
        fs: S3 filesystem client
        source: Local path to upload
        dest: S3 destination (bucket/prefix)
        verbose: Print progress

    Returns:
        Number of files uploaded
    """
    count = 0

    if source.is_file():
        dest_key = f"{dest}/{source.name}" if not dest.endswith(source.name) else dest
        if verbose:
            print(f"  {source} -> s3://{dest_key}")
        fs.put(str(source), dest_key)
        count = 1
    else:
        for root, _, files in os.walk(source):
            for file in files:
                file_path = Path(root) / file
                relative = file_path.relative_to(source)
                dest_key = f"{dest}/{relative}"
                if verbose:
                    print(f"  {file_path} -> s3://{dest_key}")
                fs.put(str(file_path), dest_key)
                count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload local data to MinIO/S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Upload Food-101 images:
    python -m engine_comparison.cli.upload --source .data/food101 --dest bucket/images/
    
  Upload NYC Taxi data:
    python -m engine_comparison.cli.upload --source .data/nyc_taxi --dest lake/taxi/
        """,
    )
    parser.add_argument(
        "--source",
        "-s",
        required=True,
        help="Local path to upload (file or directory)",
    )
    parser.add_argument(
        "--dest",
        "-d",
        required=True,
        help="S3 destination as bucket/prefix (e.g., 'bucket/images/')",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("AWS_ENDPOINT_URL", "http://minio:9000"),
        help="S3 endpoint URL (default: http://minio:9000)",
    )
    parser.add_argument(
        "--access-key",
        default=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        help="S3 access key",
    )
    parser.add_argument(
        "--secret-key",
        default=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        help="S3 secret key",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Error: Source path does not exist: {source}")
        raise SystemExit(1)

    print(f"Uploading {source} to s3://{args.dest} ...")

    fs = get_s3_client(
        endpoint_url=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
    )

    # Normalize dest path (remove trailing slashes)
    dest = args.dest.rstrip("/")

    count = upload_path(fs, source, dest, verbose=not args.quiet)

    print(f"✓ Uploaded {count} file(s) to s3://{args.dest}")


if __name__ == "__main__":
    main()
