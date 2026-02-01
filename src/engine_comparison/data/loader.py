#!/usr/bin/env python3
"""
Data Loader — Downloads and caches real-world benchmark datasets.
=================================================================

Datasets:
  1. NYC Yellow Taxi Trip Records  (~45 MB Parquet per month, ~2.9M rows)
     Source: NYC Taxi & Limousine Commission (TLC)
     URL:    https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

  2. NYC Taxi Zone Lookup           (~12 KB CSV, 265 rows)
     Source: NYC TLC

  3. Food-101 Images                (real food photos, configurable count)
     Source: ETH Zurich / Hugging Face — ethz/food101

All data is cached in .data/ so subsequent runs are instant.

Usage:
  # Pre-download everything (recommended before a live demo):
  uv run python -m engine_comparison.data.loader

  # Or just run a benchmark script — it auto-downloads on first run.
"""

from __future__ import annotations

import csv
import gc
import os
import urllib.request
from pathlib import Path

# Disable HF Hub's telemetry to prevent background threads
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import pyarrow.parquet as pq
from datasets import load_dataset
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from engine_comparison.constants import (
    DATA_DIR,
    DEFAULT_N_IMAGES,
    DEFAULT_TAXI_MONTH,
    DEFAULT_TAXI_YEAR,
    TAXI_TRIPS_URL,
    TAXI_ZONES_URL,
)

console = Console()


# ---------------------------------------------------------------------------
# NYC Taxi data
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, label: str = "Downloading") -> None:
    """Download a file with a Rich progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url, headers={"User-Agent": "engine-wars-demo/0.1"})
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", 0))

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(label, total=total or None)
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.advance(task, len(chunk))


def load_nyc_taxi(
    year: int = DEFAULT_TAXI_YEAR, month: int = DEFAULT_TAXI_MONTH
) -> tuple[Path, Path]:
    """
    Download NYC Yellow Taxi trip data (Parquet) and zone lookup (CSV).

    Returns (trips_parquet_path, zones_csv_path).
    """
    taxi_dir = DATA_DIR / "nyc_taxi"
    taxi_dir.mkdir(parents=True, exist_ok=True)

    # --- Trip records ---
    trips_path = taxi_dir / f"yellow_tripdata_{year}-{month:02d}.parquet"
    if not trips_path.exists():
        url = TAXI_TRIPS_URL.format(year=year, month=month)
        console.print(
            f"\n[bold cyan]Downloading NYC Yellow Taxi trips ({year}-{month:02d})...[/]"
        )
        _download(url, trips_path, f"yellow_tripdata_{year}-{month:02d}.parquet")
        size_mb = trips_path.stat().st_size / (1024 * 1024)
        console.print(f"  → Saved: [green]{size_mb:.1f} MB[/]")
    else:
        size_mb = trips_path.stat().st_size / (1024 * 1024)
        console.print(f"  [dim]Cached: {trips_path.name} ({size_mb:.1f} MB)[/]")

    # --- Zone lookup ---
    zones_path = taxi_dir / "taxi_zone_lookup.csv"
    if not zones_path.exists():
        console.print("[bold cyan]Downloading taxi zone lookup...[/]")
        _download(TAXI_ZONES_URL, zones_path, "taxi_zone_lookup.csv")
    else:
        console.print(f"  [dim]Cached: {zones_path.name}[/]")

    return trips_path, zones_path


# ---------------------------------------------------------------------------
# Food-101 images
# ---------------------------------------------------------------------------


def load_food101_images(n_images: int = DEFAULT_N_IMAGES) -> Path:
    """
    Download real food photos from the Food-101 dataset (ETH Zurich).

    Uses HuggingFace `datasets` to stream images and save as JPEGs.
    Returns the directory containing the image files.
    """
    images_dir = DATA_DIR / "food101"
    images_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(images_dir.glob("*.jpg"))
    if len(existing) >= n_images:
        console.print(
            f"  [dim]Cached: {len(existing)} Food-101 images in {images_dir}[/]"
        )
        return images_dir

    console.print(
        f"\n[bold cyan]Downloading {n_images} Food-101 images from Hugging Face...[/]"
    )

    # Stream mode: downloads only what we need, no full dataset required
    ds = load_dataset(
        "ethz/food101",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} images"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Saving images", total=n_images)
        for i, example in enumerate(ds):
            if i >= n_images:
                break
            path = images_dir / f"food_{i:05d}.jpg"
            if not path.exists():
                img = example["image"]
                # Convert to RGB if needed (some images might be RGBA/palette)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(path, "JPEG", quality=90)
            progress.advance(task)

    # Clean up HF Hub connections to prevent hanging
    del ds
    gc.collect()

    saved = list(images_dir.glob("*.jpg"))
    total_mb = sum(p.stat().st_size for p in saved) / (1024 * 1024)
    console.print(f"  → {len(saved)} images saved ({total_mb:.1f} MB total)\n")
    return images_dir


# ---------------------------------------------------------------------------
# CLI: pre-download everything
# ---------------------------------------------------------------------------


def main():
    console.print("\n[bold white on blue] Engine Wars — Data Downloader [/]\n")
    console.print("Downloading all benchmark datasets...\n")

    trips_path, zones_path = load_nyc_taxi()

    # Show row count
    meta = pq.read_metadata(str(trips_path))
    console.print(
        f"  → Taxi trips: [green]{meta.num_rows:,}[/] rows × {meta.num_columns} cols"
    )

    with open(zones_path) as f:
        n_zones = sum(1 for _ in csv.reader(f)) - 1
    console.print(f"  → Taxi zones: [green]{n_zones}[/] rows\n")

    images_dir = load_food101_images(n_images=DEFAULT_N_IMAGES)
    n_images = len(list(images_dir.glob("*.jpg")))
    console.print(f"  → Food-101:   [green]{n_images}[/] images\n")

    console.print("[bold green]✓ All data ready! Run the benchmarks:[/]")
    console.print("  uv run python -m engine_comparison.benchmarks.tabular")
    console.print("  uv run python -m engine_comparison.benchmarks.multimodal\n")


if __name__ == "__main__":
    main()
