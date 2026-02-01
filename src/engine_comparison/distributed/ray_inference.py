#!/usr/bin/env python3
"""
Engine Wars — Ray Data: Distributed GPU Inference on Real Images
=================================================================
Batch image classification using Ray Data's streaming actor pools.

Architecture:
  S3 Images → CPU Decode → GPU Actor Pool (model loaded once) → Parquet

Key points:
  - ActorPoolStrategy: persistent GPU workers, model loaded once per actor
  - Streaming execution: bounded memory, no full materialization
  - Backpressure: reads throttled to match GPU throughput

Requirements:
  - Ray cluster with GPU nodes (e.g., g6.8xlarge)
  - Images on S3 (or use Food-101 from HuggingFace)

Usage:
  RAY_ADDRESS=auto python ray_inference.py --input s3://bucket/images/
"""

from __future__ import annotations

import argparse
import time

import ray


class ImageClassifier:
    """Stateful GPU actor — model loaded ONCE, reused for all batches."""

    def __init__(self):
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights).to(self.device).eval()
        self.preprocess = self.weights.transforms()
        self.categories = self.weights.meta["categories"]
        print(f"[ImageClassifier] ResNet-50 loaded on {self.device}")

    def __call__(self, batch: dict) -> dict:
        import torch

        tensors = torch.stack(
            [
                self.preprocess(torch.from_numpy(img).permute(2, 0, 1))
                for img in batch["image"]
            ]
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(tensors)

        top_idx = logits.argmax(dim=1).cpu().numpy()
        return {
            "prediction": [self.categories[i] for i in top_idx],
            "confidence": logits.softmax(dim=1).max(dim=1).values.cpu().numpy(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="s3://bucket/images/")
    parser.add_argument("--output", default="s3://bucket/predictions/")
    parser.add_argument("--gpu-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ray.init()
    t0 = time.perf_counter()

    ds = ray.data.read_images(args.input)
    predictions = ds.map_batches(
        ImageClassifier,
        compute=ray.data.ActorPoolStrategy(size=args.gpu_workers),
        num_gpus=1,
        batch_size=args.batch_size,
    )
    predictions.write_parquet(args.output)

    print(f"\n✓ Complete in {time.perf_counter() - t0:.1f}s → {args.output}")
    ray.shutdown()


if __name__ == "__main__":
    main()
