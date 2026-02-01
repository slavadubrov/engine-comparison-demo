#!/usr/bin/env python3
"""
Engine Wars — Single-Node Tabular Benchmark (NYC Taxi Data)
============================================================
Downloads real NYC Yellow Taxi trip records (~2.9M rows) and benchmarks
identical analytical queries across four engines:

  Pandas · Polars · Apache DataFusion · Daft

Queries:
  1. Read Parquet      — full scan of ~45 MB trip data
  2. Filter            — long-distance, high-fare trips (distance > 5mi, fare > $30)
  3. GroupBy + Agg     — revenue breakdown by payment type
  4. Join              — enrich trips with pickup borough/zone names
  5. ETL Pipeline      — filter → join → aggregate by borough → rank by revenue

Usage:
    uv run python bench_tabular.py                    # defaults (Jan 2024)
    uv run python bench_tabular.py --year 2023 --month 6
    uv run python bench_tabular.py --runs 5           # more timing precision
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from data_loader import load_nyc_taxi

console = Console()


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


def timeit(fn, n_runs: int = 3) -> float:
    """Run fn() n_runs times, return median elapsed seconds."""
    times: list[float] = []
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
    return sorted(times)[len(times) // 2]


# ---------------------------------------------------------------------------
# Benchmarks: Pandas
# ---------------------------------------------------------------------------


def bench_pandas(trips_path: str, zones_path: str, n_runs: int) -> dict:
    import pandas as pd

    results = {}

    # Read
    results["Read Parquet"] = timeit(lambda: pd.read_parquet(trips_path), n_runs)

    df = pd.read_parquet(trips_path)
    zones = pd.read_csv(zones_path)

    # Filter: long-distance, high-fare trips
    results["Filter"] = timeit(
        lambda: df[(df["trip_distance"] > 5.0) & (df["fare_amount"] > 30.0)],
        n_runs,
    )

    # GroupBy + Agg: revenue by payment type
    results["GroupBy + Agg"] = timeit(
        lambda: df.groupby("payment_type").agg(
            trip_count=("VendorID", "count"),
            total_revenue=("total_amount", "sum"),
            avg_fare=("fare_amount", "mean"),
            avg_tip=("tip_amount", "mean"),
        ),
        n_runs,
    )

    # Join: enrich with pickup zone names
    results["Join"] = timeit(
        lambda: df.merge(
            zones, left_on="PULocationID", right_on="LocationID", how="inner"
        ),
        n_runs,
    )

    # ETL Pipeline: filter → join → groupby borough → sort
    def etl_pipeline():
        (
            df[df["fare_amount"] > 10.0]
            .merge(zones, left_on="PULocationID", right_on="LocationID", how="inner")
            .groupby(["Borough", "Zone"])
            .agg(
                revenue=("total_amount", "sum"),
                trips=("VendorID", "count"),
                avg_tip_pct=(
                    "tip_amount",
                    lambda x: (
                        x / df.loc[x.index, "fare_amount"].replace(0, np.nan)
                    ).mean(),
                ),
            )
            .sort_values("revenue", ascending=False)
            .head(20)
        )

    # Simpler pipeline that avoids lambda issues
    def etl_pipeline_safe():
        merged = df[df["fare_amount"] > 10.0].merge(
            zones, left_on="PULocationID", right_on="LocationID", how="inner"
        )
        result = (
            merged.groupby(["Borough", "Zone"])
            .agg(
                revenue=("total_amount", "sum"),
                trips=("VendorID", "count"),
                avg_tip=("tip_amount", "mean"),
            )
            .sort_values("revenue", ascending=False)
            .head(20)
        )
        return result

    results["ETL Pipeline"] = timeit(etl_pipeline_safe, n_runs)
    return results


# ---------------------------------------------------------------------------
# Benchmarks: Polars
# ---------------------------------------------------------------------------


def bench_polars(trips_path: str, zones_path: str, n_runs: int) -> dict:
    import polars as pl

    results = {}

    # Read
    results["Read Parquet"] = timeit(lambda: pl.read_parquet(trips_path), n_runs)

    # Filter (lazy — predicate pushdown into Parquet reader)
    results["Filter"] = timeit(
        lambda: (
            pl.scan_parquet(trips_path)
            .filter((pl.col("trip_distance") > 5.0) & (pl.col("fare_amount") > 30.0))
            .collect()
        ),
        n_runs,
    )

    # GroupBy + Agg (lazy)
    results["GroupBy + Agg"] = timeit(
        lambda: (
            pl.scan_parquet(trips_path)
            .group_by("payment_type")
            .agg(
                pl.len().alias("trip_count"),
                pl.col("total_amount").sum().alias("total_revenue"),
                pl.col("fare_amount").mean().alias("avg_fare"),
                pl.col("tip_amount").mean().alias("avg_tip"),
            )
            .collect()
        ),
        n_runs,
    )

    # Join (lazy scan both sides)
    results["Join"] = timeit(
        lambda: (
            pl.scan_parquet(trips_path)
            .join(
                pl.scan_csv(zones_path),
                left_on="PULocationID",
                right_on="LocationID",
                how="inner",
            )
            .collect()
        ),
        n_runs,
    )

    # ETL Pipeline (fully lazy — single optimized query plan)
    results["ETL Pipeline"] = timeit(
        lambda: (
            pl.scan_parquet(trips_path)
            .filter(pl.col("fare_amount") > 10.0)
            .join(
                pl.scan_csv(zones_path),
                left_on="PULocationID",
                right_on="LocationID",
                how="inner",
            )
            .group_by("Borough", "Zone")
            .agg(
                pl.col("total_amount").sum().alias("revenue"),
                pl.len().alias("trips"),
                pl.col("tip_amount").mean().alias("avg_tip"),
            )
            .sort("revenue", descending=True)
            .head(20)
            .collect()
        ),
        n_runs,
    )

    return results


# ---------------------------------------------------------------------------
# Benchmarks: DataFusion
# ---------------------------------------------------------------------------


def bench_datafusion(trips_path: str, zones_path: str, n_runs: int) -> dict:
    from datafusion import SessionContext

    results = {}

    # Read
    def df_read():
        ctx = SessionContext()
        ctx.register_parquet("trips", trips_path)
        ctx.sql("SELECT * FROM trips").to_arrow_table()

    results["Read Parquet"] = timeit(df_read, n_runs)

    # Filter
    def df_filter():
        ctx = SessionContext()
        ctx.register_parquet("trips", trips_path)
        ctx.sql("""
            SELECT *
            FROM trips
            WHERE trip_distance > 5.0 AND fare_amount > 30.0
        """).to_arrow_table()

    results["Filter"] = timeit(df_filter, n_runs)

    # GroupBy + Agg
    def df_groupby():
        ctx = SessionContext()
        ctx.register_parquet("trips", trips_path)
        ctx.sql("""
            SELECT payment_type,
                   COUNT(*)            AS trip_count,
                   SUM(total_amount)   AS total_revenue,
                   AVG(fare_amount)    AS avg_fare,
                   AVG(tip_amount)     AS avg_tip
            FROM trips
            GROUP BY payment_type
        """).to_arrow_table()

    results["GroupBy + Agg"] = timeit(df_groupby, n_runs)

    # Join
    def df_join():
        ctx = SessionContext()
        ctx.register_parquet("trips", trips_path)
        ctx.register_csv("zones", zones_path)
        ctx.sql("""
            SELECT t.*, z."Borough", z."Zone"
            FROM trips t
            INNER JOIN zones z ON t."PULocationID" = z."LocationID"
        """).to_arrow_table()

    results["Join"] = timeit(df_join, n_runs)

    # ETL Pipeline (single SQL query — fully optimized)
    def df_etl():
        ctx = SessionContext()
        ctx.register_parquet("trips", trips_path)
        ctx.register_csv("zones", zones_path)
        ctx.sql("""
            SELECT z."Borough",
                   z."Zone",
                   SUM(t.total_amount) AS revenue,
                   COUNT(*)            AS trips,
                   AVG(t.tip_amount)   AS avg_tip
            FROM trips t
            INNER JOIN zones z ON t."PULocationID" = z."LocationID"
            WHERE t.fare_amount > 10.0
            GROUP BY z."Borough", z."Zone"
            ORDER BY revenue DESC
            LIMIT 20
        """).to_arrow_table()

    results["ETL Pipeline"] = timeit(df_etl, n_runs)
    return results


# ---------------------------------------------------------------------------
# Benchmarks: Daft
# ---------------------------------------------------------------------------


def bench_daft(trips_path: str, zones_path: str, n_runs: int) -> dict:
    import daft
    from daft import col

    results = {}

    # Read
    results["Read Parquet"] = timeit(
        lambda: daft.read_parquet(trips_path).collect(), n_runs
    )

    # Filter
    results["Filter"] = timeit(
        lambda: (
            daft.read_parquet(trips_path)
            .where(col("trip_distance") > 5.0)
            .where(col("fare_amount") > 30.0)
            .collect()
        ),
        n_runs,
    )

    # GroupBy + Agg
    results["GroupBy + Agg"] = timeit(
        lambda: (
            daft.read_parquet(trips_path)
            .groupby("payment_type")
            .agg(
                col("total_amount").sum().alias("total_revenue"),
                col("fare_amount").mean().alias("avg_fare"),
                col("tip_amount").mean().alias("avg_tip"),
                col("VendorID").count().alias("trip_count"),
            )
            .collect()
        ),
        n_runs,
    )

    # Join
    results["Join"] = timeit(
        lambda: (
            daft.read_parquet(trips_path)
            .join(
                daft.read_csv(zones_path),
                left_on="PULocationID",
                right_on="LocationID",
            )
            .collect()
        ),
        n_runs,
    )

    # ETL Pipeline
    results["ETL Pipeline"] = timeit(
        lambda: (
            daft.read_parquet(trips_path)
            .where(col("fare_amount") > 10.0)
            .join(
                daft.read_csv(zones_path),
                left_on="PULocationID",
                right_on="LocationID",
            )
            .groupby("Borough", "Zone")
            .agg(
                col("total_amount").sum().alias("revenue"),
                col("tip_amount").mean().alias("avg_tip"),
                col("VendorID").count().alias("trips"),
            )
            .sort("revenue", desc=True)
            .limit(20)
            .collect()
        ),
        n_runs,
    )

    return results


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

OPERATIONS = ["Read Parquet", "Filter", "GroupBy + Agg", "Join", "ETL Pipeline"]
ENGINES = ["Pandas", "Polars", "DataFusion", "Daft"]
ENGINE_COLORS = {
    "Pandas": "red",
    "Polars": "blue",
    "DataFusion": "magenta",
    "Daft": "green",
}


def render_table(all_results: dict[str, dict]) -> None:
    """Print a Rich comparison table with speedup multipliers."""
    table = Table(
        title="⚡ Engine Wars — NYC Taxi Benchmark Results",
        show_lines=True,
        title_style="bold white on blue",
        padding=(0, 1),
    )
    table.add_column("Operation", style="bold", min_width=16)
    for eng in ENGINES:
        table.add_column(eng, justify="right", min_width=16)

    pandas_results = all_results.get("Pandas", {})

    for op in OPERATIONS:
        row = [op]
        for eng in ENGINES:
            t = all_results.get(eng, {}).get(op)
            if t is None:
                row.append("[dim]—[/dim]")
                continue

            pandas_t = pandas_results.get(op)
            if eng == "Pandas" or pandas_t is None or pandas_t == 0:
                row.append(f"[{ENGINE_COLORS[eng]}]{t:.3f}s[/]")
            else:
                speedup = pandas_t / t
                row.append(
                    f"[{ENGINE_COLORS[eng]}]{t:.3f}s[/] [bold green]{speedup:.1f}×[/]"
                )
        table.add_row(*row)

    # Summary row
    table.add_section()
    summary_row = ["[bold]Total"]
    for eng in ENGINES:
        total = sum(all_results.get(eng, {}).get(op, 0) for op in OPERATIONS)
        pandas_total = sum(pandas_results.get(op, 0) for op in OPERATIONS)
        if eng == "Pandas":
            summary_row.append(f"[bold red]{total:.2f}s[/]")
        elif pandas_total > 0 and total > 0:
            speedup = pandas_total / total
            summary_row.append(
                f"[bold {ENGINE_COLORS[eng]}]{total:.2f}s[/] "
                f"[bold green]{speedup:.1f}×[/]"
            )
        else:
            summary_row.append(f"[bold]{total:.2f}s[/]")
    table.add_row(*summary_row)

    console.print(table)


def save_chart(
    all_results: dict[str, dict], output_path: str = "benchmark_results.png"
) -> None:
    """Generate a grouped bar chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(OPERATIONS))
    n_engines = len(ENGINES)
    width = 0.18
    colors = ["#e74c3c", "#3498db", "#9b59b6", "#2ecc71"]

    for i, eng in enumerate(ENGINES):
        times = [all_results.get(eng, {}).get(op, 0) for op in OPERATIONS]
        offset = (i - n_engines / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            times,
            width,
            label=eng,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{t:.2f}s",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

    ax.set_xlabel("Operation", fontsize=12, fontweight="bold")
    ax.set_ylabel("Time (seconds, lower is better)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Engine Wars — NYC Yellow Taxi Benchmark (Single Node)",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(OPERATIONS, fontsize=10)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    console.print(f"\n[bold green]Chart saved → {output_path}[/]\n")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Engine Wars — NYC Taxi Tabular Benchmark"
    )
    parser.add_argument(
        "--year", type=int, default=2024, help="Taxi data year (default: 2024)"
    )
    parser.add_argument(
        "--month", type=int, default=1, help="Taxi data month (default: 1)"
    )
    parser.add_argument(
        "--runs", type=int, default=3, help="Timing runs per operation (default: 3)"
    )
    args = parser.parse_args()

    # Download / load data
    trips_path, zones_path = load_nyc_taxi(args.year, args.month)
    trips_str = str(trips_path)
    zones_str = str(zones_path)

    # Dataset info
    meta = pq.read_metadata(trips_str)
    size_mb = trips_path.stat().st_size / (1024 * 1024)

    console.print(
        Panel(
            f"[bold]Engine Wars — NYC Taxi Benchmark[/]\n\n"
            f"  Dataset:  NYC Yellow Taxi, {args.year}-{args.month:02d}\n"
            f"  Rows:     [cyan]{meta.num_rows:,}[/]\n"
            f"  Columns:  [cyan]{meta.num_columns}[/]\n"
            f"  Size:     [cyan]{size_mb:.1f} MB[/] (Parquet)\n"
            f"  Runs:     [cyan]{args.runs}[/] per operation (median reported)\n"
            f"  Engines:  Pandas · Polars · DataFusion · Daft\n"
            f"  CPU:      [cyan]{os.cpu_count()}[/] cores",
            title="⚡ Benchmark Configuration",
            border_style="blue",
        )
    )

    all_results: dict[str, dict] = {}

    # --- Pandas ---
    console.print("[bold red]▸ Benchmarking Pandas...[/]")
    all_results["Pandas"] = bench_pandas(trips_str, zones_str, args.runs)
    console.print("  ✓ Pandas done\n")

    # --- Polars ---
    console.print("[bold blue]▸ Benchmarking Polars...[/]")
    all_results["Polars"] = bench_polars(trips_str, zones_str, args.runs)
    console.print("  ✓ Polars done\n")

    # --- DataFusion ---
    console.print("[bold magenta]▸ Benchmarking DataFusion...[/]")
    all_results["DataFusion"] = bench_datafusion(trips_str, zones_str, args.runs)
    console.print("  ✓ DataFusion done\n")

    # --- Daft ---
    console.print("[bold green]▸ Benchmarking Daft...[/]")
    all_results["Daft"] = bench_daft(trips_str, zones_str, args.runs)
    console.print("  ✓ Daft done\n")

    # --- Results ---
    render_table(all_results)
    save_chart(all_results)


if __name__ == "__main__":
    main()
