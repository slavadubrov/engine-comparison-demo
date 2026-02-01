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
    uv run python bench_multimodal.py                    # 500 images
    uv run python bench_multimodal.py --images 1000
    uv run python bench_multimodal.py --images 200       # quick test
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from data_loader import load_food101_images

console = Console()

TARGET_SIZE = (224, 224)  # Standard vision model input (ResNet, ViT, etc.)


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
    import pandas as pd

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
        lambda img: img.resize(TARGET_SIZE, Image.LANCZOS)
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
    """
    import daft
    from daft import col

    results: dict[str, float] = {}
    glob_pattern = str(image_dir / "*.jpg")

    # --- Load + Decode images ---
    gc.collect()
    t0 = time.perf_counter()

    df = (
        daft.from_glob_path(glob_pattern)
        .limit(n_images)
        .with_column("image", col("path").download().decode_image())
    )
    df.collect()

    results["Load Images"] = time.perf_counter() - t0

    # --- Resize ---
    gc.collect()
    t0 = time.perf_counter()

    df_resized = (
        daft.from_glob_path(glob_pattern)
        .limit(n_images)
        .with_column("image", col("path").download().decode_image())
        .with_column("resized", col("image").resize(TARGET_SIZE[0], TARGET_SIZE[1]))
    )
    df_resized.collect()

    results["Resize 224×224"] = time.perf_counter() - t0

    # --- Full pipeline: load → decode → resize (end-to-end) ---
    gc.collect()
    t0 = time.perf_counter()

    df_full = (
        daft.from_glob_path(glob_pattern)
        .limit(n_images)
        .with_column("image", col("path").download().decode_image())
        .with_column("resized", col("image").resize(TARGET_SIZE[0], TARGET_SIZE[1]))
    )
    result = df_full.collect()

    results["Total Pipeline"] = time.perf_counter() - t0

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
    output_path: str = "multimodal_results.png",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    console.print(f"[bold green]Chart saved → {output_path}[/]\n")
    plt.close(fig)


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
        default=5000,
        help="Number of Food-101 images to benchmark (default: 5000)",
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

    console.print(
        Panel(
            f"[bold]Engine Wars — Multimodal Image Benchmark[/]\n\n"
            f"  Dataset:   Food-101 (ETH Zurich / Hugging Face)\n"
            f"  Images:    [cyan]{n}[/] real food photos\n"
            f"  Samples:   {', '.join(sample_sizes[:3])} ...\n"
            f"  Target:    [cyan]{TARGET_SIZE[0]}×{TARGET_SIZE[1]}[/] px\n"
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


if __name__ == "__main__":
    main()
