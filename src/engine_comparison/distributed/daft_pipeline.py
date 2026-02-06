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
import numpy as np
from daft import col


@daft.cls
class ImageEmbedder:
    """GPU-bound: CLIP model loaded once, encode batches of images."""

    def __init__(self):
        import logging

        import torch

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

    @daft.method.batch(
        return_dtype=daft.DataType.fixed_size_list(daft.DataType.float32(), 512)
    )
    def __call__(self, image_bytes_col):
        import io

        import numpy as np
        import torch
        from PIL import Image

        # Default embedding for failures (512-dim zero vector)
        default_embedding = np.zeros(512, dtype=np.float32)

        embeddings = []
        for img_bytes in image_bytes_col.to_pylist():
            if img_bytes is None:
                embeddings.append(default_embedding)
                continue
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                # Ensure we get exactly 512 float32 values
                emb_array = features[0].cpu().numpy().astype(np.float32)
                assert emb_array.shape == (512,), (
                    f"Expected 512-dim embedding, got {emb_array.shape}"
                )
                embeddings.append(emb_array)
            except Exception as e:
                embeddings.append(default_embedding)

        # Convert to 2D numpy array for Daft
        return np.array(embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="s3://bucket/image_metadata.parquet")
    parser.add_argument("--output", default="s3://bucket/embeddings/")
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

    elapsed = time.perf_counter() - t0

    # --- Result summary ---
    results = daft.read_parquet(args.output, io_config=io_config)
    total = results.count_rows()
    print(f"\n✓ Complete in {elapsed:.1f}s → {args.output}")
    print(f"  Total rows embedded: {total:,}")

    print("\n── Sample Rows ──")
    sample = results.limit(5).to_pandas()
    for _, row in sample.iterrows():
        emb = row["embedding"]
        emb_preview = ", ".join(f"{v:.4f}" for v in emb[:5])
        print(f"  {row['image_url'][:60]}  [{emb_preview}, ...]")

    # Embedding stats

    all_emb = np.array(results.select("embedding").to_pandas()["embedding"].tolist())
    dim = all_emb.shape[1]
    nonzero_rate = (all_emb != 0).mean()
    print(f"\n── Embedding Stats ──")
    print(f"  dimensionality: {dim}")
    print(f"  non-zero rate:  {nonzero_rate:.2%}")


if __name__ == "__main__":
    main()
