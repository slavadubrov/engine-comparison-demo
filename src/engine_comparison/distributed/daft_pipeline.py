#!/usr/bin/env python3
"""
Engine Wars — Daft: Distributed Image Embedding Pipeline
==========================================================
S3 Images → Download → GPU Embed (CLIP) → Parquet (streaming, Rust I/O)

Key points:
  - Class UDF: model loaded once per worker, reused across batches
  - url.download(): parallel Rust I/O, not Python requests
  - Memory bounded: Swordfish streams in small batches
  - Same code: local laptop → Ray cluster → Daft Cloud

Usage:
  DAFT_RUNNER=ray python daft_pipeline.py --input s3://bucket/image_metadata.parquet
"""

from __future__ import annotations

import argparse
import os
import time

import daft
from daft import col


@daft.cls
class ImageEmbedder:
    """GPU-bound: CLIP model loaded once, encode batches of images."""

    def __init__(self):
        import torch
        import logging

        # Suppress verbose transformers logging
        logging.getLogger("transformers").setLevel(logging.ERROR)

        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            # Fallback for some environments or versions
            from transformers.models.clip.modeling_clip import CLIPModel
            from transformers.models.clip.processing_clip import CLIPProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    @daft.method.batch(return_dtype=daft.DataType.python())
    def __call__(self, image_bytes_col):
        import io

        import torch
        from PIL import Image

        embeddings = []
        for img_bytes in image_bytes_col.to_pylist():
            if img_bytes is None:
                embeddings.append([0.0] * 512)  # CLIP base produces 512-dim embeddings
                continue
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                emb = features[0].cpu().numpy().tolist()
                embeddings.append(emb)
            except Exception:
                embeddings.append([0.0] * 512)
        return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="s3://bucket/image_metadata.parquet")
    parser.add_argument("--output", default="s3://output/embeddings/")
    args = parser.parse_args()

    t0 = time.perf_counter()

    # Configure S3/MinIO endpoint for Daft I/O
    endpoint_url = os.environ.get("AWS_ENDPOINT_URL", "http://minio:9000")
    io_config = daft.io.IOConfig(
        s3=daft.io.S3Config(
            endpoint_url=endpoint_url,
            key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
            access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            region_name="us-east-1",  # MinIO default
        )
    )
    daft.set_planning_config(default_io_config=io_config)

    # Create instance of daft.cls
    embedder = ImageEmbedder()

    df = (
        daft.read_parquet(args.input)
        .into_partitions(1)  # Limit concurrency to 2 workers for single GPU
        .with_column("image_bytes", col("image_url").download())
        .with_column("embedding", embedder(col("image_bytes")))
        .exclude("image_bytes")
    )
    df.write_parquet(args.output)

    print(f"\n✓ Complete in {time.perf_counter() - t0:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
