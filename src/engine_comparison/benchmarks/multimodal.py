#!/usr/bin/env python3
"""
Engine Wars — Multimodal Image Benchmark (Food-101 Dataset)
=============================================================
Downloads real food photos from ETH Zurich's Food-101 dataset, then
benchmarks image preprocessing pipelines:

  Pandas + Pillow  — sequential Python (GIL-bound, one image at a time)
  Daft (Rust)      — parallel native image ops (bypasses GIL entirely)

This demonstrates the "Multimodal Shift" from the Engine Wars article:
traditional DataFrame engines treat images as opaque blobs and delegate
to sequential Python. Daft runs decode/resize/encode in parallel Rust
threads, achieving dramatic speedups on multicore hardware.

Note: Polars and DataFusion are NOT included because they have no native
image operations. Image work would still go through sequential Python
(map_elements / UDFs), performing similarly to Pandas.

Usage:
    uv run python -m engine_comparison.benchmarks.multimodal
    uv run python -m engine_comparison.benchmarks.multimodal --images 1000
    uv run python -m engine_comparison.benchmarks.multimodal --images 200
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import daft
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from daft import col
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from engine_comparison.constants import (
    BENCHMARKS_OUTPUT_DIR,
    DEFAULT_N_IMAGES,
    MULTIMODAL_CHART_OUTPUT,
    MULTIMODAL_JSON_OUTPUT,
    TARGET_IMAGE_SIZE,
)
from engine_comparison.data.loader import load_food101_images

console = Console()

# Use non-interactive backend for chart generation
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Benchmark: Pandas + Pillow (sequential Python)
# ---------------------------------------------------------------------------


def bench_pandas_pillow(image_dir: Path, n_images: int) -> dict:
    """
    The traditional ML preprocessing approach:
      1. Build a DataFrame of file paths
      2. .apply(Image.open) — one image at a time, GIL-bound
      3. .apply(img.resize) — still sequential
      4. .apply(np.asarray)  — Python object overhead

    Every step is single-threaded. On a 16-core machine, 15 cores sit idle.
    """
    results: dict[str, float] = {}

    image_paths = sorted(image_dir.glob("*.jpg"))[:n_images]
    path_strings = [str(p) for p in image_paths]

    # --- Load images ---
    gc.collect()
    t0 = time.perf_counter()

    df = pd.DataFrame({"path": path_strings})
    df["image"] = df["path"].apply(lambda p: Image.open(p).copy())

    results["Load Images"] = time.perf_counter() - t0

    # --- Resize to 224×224 ---
    gc.collect()
    t0 = time.perf_counter()

    df["resized"] = df["image"].apply(
        lambda img: img.resize(TARGET_IMAGE_SIZE, Image.LANCZOS)
    )

    results["Resize 224×224"] = time.perf_counter() - t0

    # --- Convert to NumPy tensors (model input prep) ---
    gc.collect()
    t0 = time.perf_counter()

    df["tensor"] = df["resized"].apply(
        lambda img: np.asarray(img, dtype=np.float32) / 255.0
    )

    results["To Tensor"] = time.perf_counter() - t0

    # --- Total ---
    results["Total Pipeline"] = sum(results.values())

    return results


# ---------------------------------------------------------------------------
# Benchmark: Daft (native Rust image processing)
# ---------------------------------------------------------------------------


def bench_daft_native(image_dir: Path, n_images: int) -> dict:
    """
    Daft's multimodal-native approach:
      1. from_glob_path("*.jpg")  — Rust file discovery
      2. .download()              — parallel I/O in Rust threads
      3. .decode_image()          — JPEG decode in Rust (parallel)
      4. .resize(224, 224)        — resize in Rust (parallel)

    All heavy work runs in the Rust engine, completely bypassing Python's
    GIL. On a 16-core machine, all 16 cores participate.

    NOTE: Daft uses lazy execution, so load/decode/resize happen together
    in an optimized pipeline. We measure the end-to-end time only.
    Individual operation times cannot be isolated.
    """
    results: dict[str, float] = {}
    glob_pattern = str(image_dir / "*.jpg")

    # --- Full pipeline: load → decode → resize (end-to-end) ---
    # Daft's lazy execution fuses all operations together for optimal
    # performance. We can only measure the total time, not individual steps.
    gc.collect()
    t0 = time.perf_counter()

    df_full = (
        daft.from_glob_path(glob_pattern)
        .limit(n_images)
        .with_column("image", col("path").download().decode_image())
        .with_column(
            "resized", col("image").resize(TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1])
        )
    )
    df_full.collect()

    total_time = time.perf_counter() - t0

    # We report Total Pipeline only. Individual operations are fused and
    # cannot be separated in Daft's execution model.
    results["Total Pipeline"] = total_time

    return results


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

OPERATIONS = ["Load Images", "Resize 224×224", "Total Pipeline"]


def render_results(pandas_results: dict, daft_results: dict) -> None:
    table = Table(
        title="🖼  Engine Wars — Food-101 Multimodal Benchmark",
        show_lines=True,
        title_style="bold white on blue",
    )
    table.add_column("Operation", style="bold", min_width=18)
    table.add_column("Pandas + Pillow", justify="right", min_width=18)
    table.add_column("Daft (Rust)", justify="right", min_width=18)
    table.add_column("Speedup", justify="right", min_width=10)

    for op in OPERATIONS:
        p_time = pandas_results.get(op)
        d_time = daft_results.get(op)

        if p_time and d_time and d_time > 0:
            speedup = p_time / d_time
            speedup_str = f"[bold green]{speedup:.1f}×[/]"
        else:
            speedup_str = "[dim]—[/]"

        p_str = f"[red]{p_time:.3f}s[/]" if p_time else "[dim]—[/]"
        d_str = f"[green]{d_time:.3f}s[/]" if d_time else "[dim]—[/]"

        table.add_row(op, p_str, d_str, speedup_str)

    console.print(table)

    # Print the key insight
    console.print(
        "\n[bold yellow]Key Insight:[/] Polars and DataFusion are excluded "
        "because they have no native image operations — image work would "
        "still go through sequential Python, performing similarly to Pandas.\n"
        "The bottleneck is [underline]not[/underline] the DataFrame layer; "
        "it's the Python GIL.\n"
    )


def save_chart(
    pandas_results: dict,
    daft_results: dict,
    output_path: str = MULTIMODAL_CHART_OUTPUT,
) -> None:
    operations = ["Load Images", "Resize 224×224", "Total Pipeline"]
    p_times = [pandas_results.get(op, 0) for op in operations]
    d_times = [daft_results.get(op, 0) for op in operations]

    x = np.arange(len(operations))
    width = 0.30

    fig, ax = plt.subplots(figsize=(10, 6))
    bars_p = ax.bar(
        x - width / 2,
        p_times,
        width,
        label="Pandas + Pillow (sequential)",
        color="#e74c3c",
        edgecolor="white",
    )
    bars_d = ax.bar(
        x + width / 2,
        d_times,
        width,
        label="Daft — Rust native (parallel)",
        color="#2ecc71",
        edgecolor="white",
    )

    # Speedup annotations
    for i, (p, d) in enumerate(zip(p_times, d_times)):
        if d > 0 and p > 0:
            ax.annotate(
                f"{p / d:.1f}×",
                xy=(i, max(p, d)),
                xytext=(0, 14),
                textcoords="offset points",
                ha="center",
                fontsize=13,
                fontweight="bold",
                color="#2ecc71",
            )

    ax.set_ylabel("Time (seconds, lower is better)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Multimodal Image Processing: Food-101 Photos",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(operations, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    BENCHMARKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    console.print(f"[bold green]Chart saved → {output_path}[/]\n")
    plt.close(fig)


def save_json_report(
    pandas_results: dict,
    daft_results: dict,
    dataset_info: dict,
    output_path: str = MULTIMODAL_JSON_OUTPUT,
) -> None:
    """Save benchmark results as JSON for aggregation with Rust benchmarks."""
    report = {
        "benchmark": "multimodal",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_info,
        "results": {
            "Pandas + Pillow": pandas_results,
            "Daft": daft_results,
        },
    }
    BENCHMARKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    console.print(f"[bold green]JSON report saved → {output_path}[/]\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Engine Wars — Food-101 Multimodal Benchmark"
    )
    parser.add_argument(
        "--images",
        type=int,
        default=DEFAULT_N_IMAGES,
        help=f"Number of Food-101 images to benchmark (default: {DEFAULT_N_IMAGES})",
    )
    args = parser.parse_args()

    # Download / load images
    images_dir = load_food101_images(n_images=args.images)
    actual_count = len(list(images_dir.glob("*.jpg")))
    n = min(args.images, actual_count)

    # Sample image sizes
    sample_paths = sorted(images_dir.glob("*.jpg"))[:5]
    sample_sizes = []
    for p in sample_paths:
        img = Image.open(p)
        sample_sizes.append(f"{img.size[0]}×{img.size[1]}")

    dataset_info = {
        "name": "Food-101",
        "images": n,
        "target_size": list(TARGET_IMAGE_SIZE),
        "cpu_cores": os.cpu_count(),
    }

    console.print(
        Panel(
            f"[bold]Engine Wars — Multimodal Image Benchmark[/]\n\n"
            f"  Dataset:   Food-101 (ETH Zurich / Hugging Face)\n"
            f"  Images:    [cyan]{n}[/] real food photos\n"
            f"  Samples:   {', '.join(sample_sizes[:3])} ...\n"
            f"  Target:    [cyan]{TARGET_IMAGE_SIZE[0]}×{TARGET_IMAGE_SIZE[1]}[/] px\n"
            f"  Engines:   Pandas + Pillow  vs.  Daft (Rust native)\n"
            f"  CPU cores: [cyan]{os.cpu_count()}[/]",
            title="🖼  Benchmark Configuration",
            border_style="blue",
        )
    )

    # --- Pandas + Pillow ---
    console.print("[bold red]▸ Benchmarking Pandas + Pillow (sequential)...[/]")
    pandas_results = bench_pandas_pillow(images_dir, n)
    console.print("  ✓ Pandas done\n")

    # --- Daft (Rust) ---
    console.print("[bold green]▸ Benchmarking Daft (Rust-native parallel)...[/]")
    daft_results = bench_daft_native(images_dir, n)
    console.print("  ✓ Daft done\n")

    # --- Results ---
    render_results(pandas_results, daft_results)
    save_chart(pandas_results, daft_results)
    save_json_report(pandas_results, daft_results, dataset_info)


if __name__ == "__main__":
    main()
