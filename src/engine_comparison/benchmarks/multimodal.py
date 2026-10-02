#!/usr/bin/env python3
"""
Engine Comparison — Multimodal Image Benchmark (Food-101 Dataset)
=============================================================
Downloads real food photos from ETH Zurich's Food-101 dataset, then
benchmarks image preprocessing pipelines:

  Pandas + Pillow  — one image at a time through DataFrame.apply()
  Daft (Rust)      — native image expressions that run in parallel

Both engines read, decode, and resize the same images to 224×224 (bilinear)
inside the timed function. The script reports the median of --runs runs and
checks that each engine produced one 224×224 RGB image per input.

Note: Polars and DataFusion are NOT included because they have no native
image operations. Image work would still go through sequential Python
(map_elements / UDFs), performing similarly to Pandas.

Usage:
    uv run python -m engine_comparison.benchmarks.multimodal
    uv run python -m engine_comparison.benchmarks.multimodal --images 1000
    uv run python -m engine_comparison.benchmarks.multimodal --images 200 --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

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
    DEFAULT_BENCHMARK_RUNS,
    DEFAULT_N_IMAGES,
    MULTIMODAL_CHART_OUTPUT,
    MULTIMODAL_JSON_OUTPUT,
    TARGET_IMAGE_SIZE,
)
from engine_comparison.benchmarks.tabular import timeit
from engine_comparison.data.loader import load_food101_images

console = Console()

# Use non-interactive backend for chart generation
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# The timed job
# ---------------------------------------------------------------------------
# Every engine gets the same list of image paths and does the same work inside
# the timed function: read each JPEG from disk, decode it to RGB, and resize it
# to 224×224 with a bilinear filter (Daft's resize filter). Each run must
# produce one 224×224 RGB image per input path; a failed image stops the run.


def bench_pandas_pillow(paths: list[str], n_runs: int) -> tuple[float, list]:
    """Pandas + Pillow: .apply() processes one image at a time."""

    def job():
        df = pd.DataFrame({"path": paths})
        df["image"] = df["path"].apply(lambda p: Image.open(p).convert("RGB"))
        df["resized"] = df["image"].apply(
            lambda img: img.resize(TARGET_IMAGE_SIZE, Image.BILINEAR)
        )
        return df["resized"].tolist()

    t, images = timeit(job, n_runs)
    shapes = [img.size[::-1] + (len(img.getbands()),) for img in images]
    return t, shapes


def bench_daft_native(paths: list[str], n_runs: int) -> tuple[float, list]:
    """Daft: read, decode, and resize with native expressions that run in
    parallel Rust threads. Lazy execution fuses the steps, so only the total
    is measured."""

    def job():
        return (
            daft.from_pydict({"path": paths})
            .with_column("image", col("path").download().decode_image(mode="RGB"))
            .with_column(
                "resized",
                col("image").resize(TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1]),
            )
            .select("resized")
            .collect()
        )

    t, df = timeit(job, n_runs)
    shapes = [np.asarray(img).shape for img in df.to_pydict()["resized"]]
    return t, shapes


def check_outputs(engine: str, shapes: list, n: int) -> None:
    w, h = TARGET_IMAGE_SIZE
    if len(shapes) != n or any(tuple(s) != (h, w, 3) for s in shapes):
        raise SystemExit(f"{engine} did not produce {n} {w}×{h} RGB images")
    console.print(f"  ✓ {engine} produced {n} {w}×{h} RGB images")


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

OPERATIONS = ["Total Pipeline"]


def render_results(pandas_results: dict, daft_results: dict) -> None:
    table = Table(
        title="🖼  Engine Comparison — Food-101 Multimodal Benchmark",
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

    console.print(
        "\n[dim]Polars and DataFusion are not included: they have no native "
        "image operations, so image work would run through Python UDFs.[/]\n"
    )


def save_chart(
    pandas_results: dict,
    daft_results: dict,
    output_path: str = MULTIMODAL_CHART_OUTPUT,
) -> None:
    operations = ["Total Pipeline"]
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
        description="Engine Comparison — Food-101 Multimodal Benchmark"
    )
    parser.add_argument(
        "--images",
        type=int,
        default=DEFAULT_N_IMAGES,
        help=f"Number of Food-101 images to benchmark (default: {DEFAULT_N_IMAGES})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_BENCHMARK_RUNS,
        help=f"Timing runs (default: {DEFAULT_BENCHMARK_RUNS}, median reported)",
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
        "runs": args.runs,
    }

    console.print(
        Panel(
            f"[bold]Engine Comparison — Multimodal Image Benchmark[/]\n\n"
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

    paths = [str(p) for p in sorted(images_dir.glob("*.jpg"))[:n]]

    # --- Pandas + Pillow ---
    console.print("[bold red]▸ Benchmarking Pandas + Pillow (sequential)...[/]")
    t, shapes = bench_pandas_pillow(paths, args.runs)
    check_outputs("Pandas + Pillow", shapes, n)
    pandas_results = {"Total Pipeline": t}

    # --- Daft (Rust) ---
    console.print("[bold green]▸ Benchmarking Daft (Rust-native parallel)...[/]")
    t, shapes = bench_daft_native(paths, args.runs)
    check_outputs("Daft", shapes, n)
    daft_results = {"Total Pipeline": t}

    # --- Results ---
    render_results(pandas_results, daft_results)
    save_chart(pandas_results, daft_results)
    save_json_report(pandas_results, daft_results, dataset_info)


if __name__ == "__main__":
    main()
