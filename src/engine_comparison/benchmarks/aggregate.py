#!/usr/bin/env python3
"""
Engine Wars — Results Aggregator
================================
Combines benchmark results from Python and Rust benchmarks,
generating unified comparison charts.

Usage:
    uv run python -m engine_comparison.benchmarks.aggregate
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from engine_comparison.constants import (
    BENCHMARKS_OUTPUT_DIR,
    COMBINED_MULTIMODAL_CHART,
    COMBINED_TABULAR_CHART,
    MULTIMODAL_JSON_OUTPUT,
    RUST_MULTIMODAL_JSON_OUTPUT,
    RUST_TABULAR_JSON_OUTPUT,
    TABULAR_JSON_OUTPUT,
)

console = Console()
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Tabular Aggregation
# ---------------------------------------------------------------------------

TABULAR_OPERATIONS = ["Read Parquet", "Filter", "GroupBy + Agg", "Join", "ETL Pipeline"]
TABULAR_ENGINES = ["Pandas", "Polars", "DataFusion", "Daft", "Polars-rs"]
TABULAR_COLORS = ["#e74c3c", "#3498db", "#9b59b6", "#2ecc71", "#f39c12"]


def aggregate_tabular() -> dict[str, dict] | None:
    """Load and merge Python + Rust tabular results."""
    results: dict[str, dict] = {}

    # Python results
    if TABULAR_JSON_OUTPUT.exists():
        with open(TABULAR_JSON_OUTPUT) as f:
            data = json.load(f)
            results.update(data.get("results", {}))
        console.print(f"[green]✓[/] Loaded {TABULAR_JSON_OUTPUT}")
    else:
        console.print(f"[yellow]⚠[/] {TABULAR_JSON_OUTPUT} not found")

    # Rust results
    if RUST_TABULAR_JSON_OUTPUT.exists():
        with open(RUST_TABULAR_JSON_OUTPUT) as f:
            data = json.load(f)
            results.update(data.get("results", {}))
        console.print(f"[green]✓[/] Loaded {RUST_TABULAR_JSON_OUTPUT}")
    else:
        console.print(f"[yellow]⚠[/] {RUST_TABULAR_JSON_OUTPUT} not found")

    return results if results else None


def render_tabular_table(results: dict[str, dict]) -> None:
    """Print a Rich comparison table for tabular results."""
    table = Table(
        title="⚡ Combined Tabular Benchmark (Python + Rust)",
        show_lines=True,
        title_style="bold white on blue",
    )
    table.add_column("Operation", style="bold", min_width=16)
    for eng in TABULAR_ENGINES:
        table.add_column(eng, justify="right", min_width=12)

    pandas_results = results.get("Pandas", {})

    for op in TABULAR_OPERATIONS:
        row = [op]
        for eng in TABULAR_ENGINES:
            t = results.get(eng, {}).get(op)
            if t is None:
                row.append("[dim]—[/dim]")
                continue

            pandas_t = pandas_results.get(op)
            if eng == "Pandas" or pandas_t is None or pandas_t == 0:
                row.append(f"{t:.3f}s")
            else:
                speedup = pandas_t / t
                row.append(f"{t:.3f}s [green]{speedup:.1f}×[/]")
        table.add_row(*row)

    console.print(table)


def save_tabular_chart(results: dict[str, dict]) -> None:
    """Generate combined tabular benchmark chart."""
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(TABULAR_OPERATIONS))
    n_engines = len(TABULAR_ENGINES)
    width = 0.15

    for i, (eng, color) in enumerate(zip(TABULAR_ENGINES, TABULAR_COLORS)):
        times = [results.get(eng, {}).get(op, 0) for op in TABULAR_OPERATIONS]
        offset = (i - n_engines / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, times, width, label=eng, color=color, edgecolor="white"
        )
        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{t:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    fontweight="bold",
                )

    ax.set_xlabel("Operation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time (seconds, lower is better)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Combined Benchmark — Python Engines + Polars-rs (Rust)",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(TABULAR_OPERATIONS, fontsize=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    BENCHMARKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMBINED_TABULAR_CHART, dpi=150, bbox_inches="tight")
    console.print(f"[bold green]Chart saved → {COMBINED_TABULAR_CHART}[/]\n")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multimodal Aggregation
# ---------------------------------------------------------------------------

MULTIMODAL_OPERATIONS = ["Load Images", "Resize 224×224", "Total Pipeline"]
MULTIMODAL_ENGINES = ["Pandas + Pillow", "Daft", "Rust image"]
MULTIMODAL_COLORS = ["#e74c3c", "#2ecc71", "#f39c12"]


def aggregate_multimodal() -> dict[str, dict] | None:
    """Load and merge Python + Rust multimodal results."""
    results: dict[str, dict] = {}

    # Python results
    if MULTIMODAL_JSON_OUTPUT.exists():
        with open(MULTIMODAL_JSON_OUTPUT) as f:
            data = json.load(f)
            results.update(data.get("results", {}))
        console.print(f"[green]✓[/] Loaded {MULTIMODAL_JSON_OUTPUT}")
    else:
        console.print(f"[yellow]⚠[/] {MULTIMODAL_JSON_OUTPUT} not found")

    # Rust results
    if RUST_MULTIMODAL_JSON_OUTPUT.exists():
        with open(RUST_MULTIMODAL_JSON_OUTPUT) as f:
            data = json.load(f)
            results.update(data.get("results", {}))
        console.print(f"[green]✓[/] Loaded {RUST_MULTIMODAL_JSON_OUTPUT}")
    else:
        console.print(f"[yellow]⚠[/] {RUST_MULTIMODAL_JSON_OUTPUT} not found")

    return results if results else None


def render_multimodal_table(results: dict[str, dict]) -> None:
    """Print a Rich comparison table for multimodal results."""
    table = Table(
        title="🖼  Combined Multimodal Benchmark (Python + Rust)",
        show_lines=True,
        title_style="bold white on blue",
    )
    table.add_column("Operation", style="bold", min_width=18)
    for eng in MULTIMODAL_ENGINES:
        table.add_column(eng, justify="right", min_width=16)
    table.add_column("Best Speedup", justify="right", min_width=12)

    pandas_results = results.get("Pandas + Pillow", {})

    for op in MULTIMODAL_OPERATIONS:
        row = [op]
        best_speedup = 1.0
        pandas_t = pandas_results.get(op, 0)

        for eng in MULTIMODAL_ENGINES:
            t = results.get(eng, {}).get(op)
            if t is None:
                row.append("[dim]—[/dim]")
                continue

            row.append(f"{t:.3f}s")
            if pandas_t > 0 and t > 0:
                speedup = pandas_t / t
                if speedup > best_speedup:
                    best_speedup = speedup

        row.append(
            f"[bold green]{best_speedup:.1f}×[/]" if best_speedup > 1 else "[dim]—[/]"
        )
        table.add_row(*row)

    console.print(table)


def save_multimodal_chart(results: dict[str, dict]) -> None:
    """Generate combined multimodal benchmark chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(MULTIMODAL_OPERATIONS))
    width = 0.25

    for i, (eng, color) in enumerate(zip(MULTIMODAL_ENGINES, MULTIMODAL_COLORS)):
        times = [results.get(eng, {}).get(op, 0) for op in MULTIMODAL_OPERATIONS]
        offset = (i - len(MULTIMODAL_ENGINES) / 2 + 0.5) * width
        ax.bar(x + offset, times, width, label=eng, color=color, edgecolor="white")

    # Add speedup annotations
    pandas_results = results.get("Pandas + Pillow", {})
    for i, op in enumerate(MULTIMODAL_OPERATIONS):
        pandas_t = pandas_results.get(op, 0)
        if pandas_t > 0:
            best_time = min(
                results.get(eng, {}).get(op, float("inf"))
                for eng in MULTIMODAL_ENGINES
                if results.get(eng, {}).get(op)
            )
            if best_time < pandas_t:
                speedup = pandas_t / best_time
                ax.annotate(
                    f"{speedup:.1f}×",
                    xy=(i, pandas_t),
                    xytext=(0, 14),
                    textcoords="offset points",
                    ha="center",
                    fontsize=12,
                    fontweight="bold",
                    color="#2ecc71",
                )

    ax.set_ylabel("Time (seconds, lower is better)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Combined Multimodal Benchmark — Python vs Rust",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(MULTIMODAL_OPERATIONS, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    BENCHMARKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(COMBINED_MULTIMODAL_CHART, dpi=150, bbox_inches="tight")
    console.print(f"[bold green]Chart saved → {COMBINED_MULTIMODAL_CHART}[/]\n")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    console.print("\n[bold blue]📊 Aggregating benchmark results...[/]\n")

    # --- Tabular ---
    tabular_results = aggregate_tabular()
    if tabular_results:
        console.print()
        render_tabular_table(tabular_results)
        save_tabular_chart(tabular_results)
    else:
        console.print("[red]✗[/] No tabular results found\n")

    # --- Multimodal ---
    multimodal_results = aggregate_multimodal()
    if multimodal_results:
        console.print()
        render_multimodal_table(multimodal_results)
        save_multimodal_chart(multimodal_results)
    else:
        console.print("[red]✗[/] No multimodal results found\n")

    console.print("[bold green]✅ Aggregation complete![/]\n")


if __name__ == "__main__":
    main()
