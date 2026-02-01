#!/usr/bin/env python3
"""
Engine Wars — Daft: Distributed Document Embedding Pipeline
=============================================================
S3 PDFs → Parse Text → GPU Embed → Parquet (all streaming, all Rust I/O)

Key points:
  - Class UDF: model loaded once per worker, reused across batches
  - url.download(): parallel Rust I/O, not Python requests
  - Memory bounded: Swordfish streams in small batches
  - Same code: local laptop → Ray cluster → Daft Cloud

Usage:
  DAFT_RUNNER=ray python daft_pipeline.py --input s3://lake/pdfs.parquet
"""

from __future__ import annotations

import argparse
import time

import daft
from daft import col


@daft.udf(return_dtype=daft.DataType.string())
def parse_pdf(pdf_bytes_col):
    """CPU-bound: extract text from PDF bytes via PyMuPDF."""
    import fitz

    results = []
    for pdf_bytes in pdf_bytes_col.to_pylist():
        if pdf_bytes is None:
            results.append(None)
            continue
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            results.append(text[:10_000] if text else None)
        except Exception:
            results.append(None)
    return results


@daft.udf(return_dtype=daft.DataType.list(daft.DataType.float32()))
class TextEmbedder:
    """GPU-bound: model loaded once, encode batches of text."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

    def __call__(self, text_col):
        texts = [t if t else "" for t in text_col.to_pylist()]
        embeddings = self.model.encode(texts, batch_size=32)
        return [emb.tolist() for emb in embeddings]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="s3://lake/pdf_metadata.parquet")
    parser.add_argument("--output", default="s3://output/embeddings/")
    args = parser.parse_args()

    t0 = time.perf_counter()

    df = (
        daft.read_parquet(args.input)
        .with_column("pdf_bytes", col("pdf_url").url.download())
        .with_column("text", parse_pdf(col("pdf_bytes")))
        .exclude("pdf_bytes")
        .with_column("embedding", TextEmbedder(col("text")))
    )
    df.write_parquet(args.output)

    print(f"\n✓ Complete in {time.perf_counter() - t0:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
