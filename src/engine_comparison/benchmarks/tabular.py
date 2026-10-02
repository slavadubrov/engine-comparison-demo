#!/usr/bin/env python3
"""
Engine Comparison — Single-Node Tabular Benchmark (NYC Taxi Data)
============================================================
Downloads NYC Yellow Taxi trip records for one year (~41M rows for 2024) and
times the same job on four engines:

  Pandas · Polars · Apache DataFusion · Daft

Job ("ETL Pipeline"): read the Parquet files and the zone CSV, keep trips with
fare_amount > 10, join zones, sum revenue by borough and zone, return the top
20. Each engine reads its input inside the timed function. The script reports
the median of --runs runs and checks that every engine returns the same rows.

Usage:
    uv run python -m engine_comparison.benchmarks.tabular
    uv run python -m engine_comparison.benchmarks.tabular --runs 5
"""

from __future__ import annotations

import argparse
import gc
import glob
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
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
from daft import col
from datafusion import SessionContext
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from engine_comparison.constants import (
    BENCHMARKS_OUTPUT_DIR,
    DATA_DIR,
    DEFAULT_BENCHMARK_RUNS,
    DEFAULT_TAXI_YEAR,
    TABULAR_CHART_OUTPUT,
    TABULAR_JSON_OUTPUT,
)
from engine_comparison.data.loader import load_nyc_taxi

console = Console()

# Use non-interactive backend for chart generation
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------


def timeit(fn, n_runs: int = DEFAULT_BENCHMARK_RUNS) -> tuple[float, object]:
    """Run fn() n_runs times. Return (median elapsed seconds, last result)."""
    times: list[float] = []
    result = None
    for _ in range(n_runs):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2], result


# ---------------------------------------------------------------------------
# The timed job
# ---------------------------------------------------------------------------
# Every engine runs the same job inside the timed function:
#   read the 12 monthly Parquet files and the zone CSV from disk
#   → keep trips with fare_amount > 10
#   → inner join zones on PULocationID = LocationID
#   → group by Borough, Zone: sum(total_amount), count, mean(tip_amount)
#   → sort by revenue descending → top 20
# Nothing is loaded before the timer starts. Each function returns the
# result as an Arrow table so main() can check that all engines agree.

TRIP_COLUMNS = ["PULocationID", "fare_amount", "total_amount", "tip_amount"]
RESULT_COLUMNS = ["Borough", "Zone", "revenue", "trips", "avg_tip"]


def bench_pandas(trips_glob: str, zones_path: str, n_runs: int):
    trip_files = sorted(glob.glob(trips_glob))

    def job():
        # Pandas has no query optimizer, so the script names the columns
        # the query needs. The lazy engines find them on their own.
        df = pd.read_parquet(trip_files, columns=TRIP_COLUMNS)
        # keep_default_na=False keeps the literal "N/A" zone names, as the
        # other engines do.
        zones = pd.read_csv(zones_path, keep_default_na=False)
        merged = df[df["fare_amount"] > 10.0].merge(
            zones, left_on="PULocationID", right_on="LocationID", how="inner"
        )
        return (
            merged.groupby(["Borough", "Zone"], as_index=False)
            .agg(
                revenue=("total_amount", "sum"),
                trips=("total_amount", "size"),
                avg_tip=("tip_amount", "mean"),
            )
            .sort_values("revenue", ascending=False)
            .head(20)
        )

    t, result = timeit(job, n_runs)
    return t, pa.Table.from_pandas(result[RESULT_COLUMNS], preserve_index=False)


def bench_polars(trips_glob: str, zones_path: str, n_runs: int):
    def job():
        return (
            pl.scan_parquet(trips_glob)
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
        )

    t, result = timeit(job, n_runs)
    return t, result.select(RESULT_COLUMNS).to_arrow()


def bench_datafusion(trips_glob: str, zones_path: str, n_runs: int):
    def job():
        ctx = SessionContext()
        ctx.register_parquet("trips", trips_glob)
        ctx.register_csv("zones", zones_path)
        return ctx.sql("""
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

    return timeit(job, n_runs)


def bench_daft(trips_glob: str, zones_path: str, n_runs: int):
    def job():
        return (
            daft.read_parquet(trips_glob)
            .where(col("fare_amount") > 10.0)
            .join(
                daft.read_csv(zones_path),
                left_on="PULocationID",
                right_on="LocationID",
            )
            .groupby("Borough", "Zone")
            .agg(
                col("total_amount").sum().alias("revenue"),
                col("total_amount").count().alias("trips"),
                col("tip_amount").mean().alias("avg_tip"),
            )
            .sort("revenue", desc=True)
            .limit(20)
            .collect()
        )

    t, result = timeit(job, n_runs)
    return t, result.select(*RESULT_COLUMNS).to_arrow()


def result_key(table: pa.Table) -> list[tuple]:
    """Rows as (Borough, Zone, trips, revenue rounded to cents), in order."""
    d = table.to_pydict()
    return [
        (b, z, int(n), round(float(r), 2))
        for b, z, n, r in zip(d["Borough"], d["Zone"], d["trips"], d["revenue"])
    ]


def check_results(results: dict[str, pa.Table]) -> None:
    """Fail if any engine returned different rows from Pandas."""
    expected = result_key(results["Pandas"])
    for eng, table in results.items():
        got = result_key(table)
        if got != expected:
            raise SystemExit(
                f"{eng} returned a different result from Pandas:\n"
                f"  Pandas: {expected[:3]} ...\n  {eng}: {got[:3]} ..."
            )
    console.print(
        f"[bold green]✓ All engines returned the same {len(expected)} rows[/]\n"
    )


# ---------------------------------------------------------------------------
# Results rendering
# ---------------------------------------------------------------------------

OPERATIONS = ["ETL Pipeline"]
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
        title="⚡ Engine Comparison — NYC Taxi Benchmark Results",
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
    all_results: dict[str, dict], output_path: str = TABULAR_CHART_OUTPUT
) -> None:
    """Generate a grouped bar chart."""
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
        "Engine Comparison — NYC Yellow Taxi Benchmark (Single Node)",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(OPERATIONS, fontsize=10)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    BENCHMARKS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    console.print(f"\n[bold green]Chart saved → {output_path}[/]\n")
    plt.close(fig)


def save_json_report(
    all_results: dict[str, dict],
    dataset_info: dict,
    output_path: str = TABULAR_JSON_OUTPUT,
) -> None:
    """Save benchmark results as JSON for aggregation with Rust benchmarks."""
    report = {
        "benchmark": "tabular",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_info,
        "results": all_results,
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
        description="Engine Comparison — NYC Taxi Tabular Benchmark"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_TAXI_YEAR,
        help=f"Taxi data year (default: {DEFAULT_TAXI_YEAR})",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_BENCHMARK_RUNS,
        help=f"Timing runs per operation (default: {DEFAULT_BENCHMARK_RUNS})",
    )
    args = parser.parse_args()

    # Download / load data
    _trips_paths, zones_path = load_nyc_taxi(args.year)
    trips_glob = str(DATA_DIR / "nyc_taxi" / f"yellow_tripdata_{args.year}-*.parquet")
    zones_str = str(zones_path)

    # Dataset info
    trip_dataset = ds.dataset(glob.glob(trips_glob))
    total_rows = sum(frag.count_rows() for frag in trip_dataset.get_fragments())
    total_cols = len(trip_dataset.schema.names)

    total_size_mb = sum(Path(f).stat().st_size for f in glob.glob(trips_glob)) / (
        1024 * 1024
    )

    dataset_info = {
        "name": "NYC Yellow Taxi",
        "year": args.year,
        "rows": total_rows,
        "columns": total_cols,
        "size_mb": round(total_size_mb, 1),
    }

    console.print(
        Panel(
            f"[bold]Engine Comparison — NYC Taxi Benchmark[/]\n\n"
            f"  Dataset:  NYC Yellow Taxi, full year {args.year} (12 months)\n"
            f"  Rows:     [cyan]{total_rows:,}[/]\n"
            f"  Columns:  [cyan]{total_cols}[/]\n"
            f"  Size:     [cyan]{total_size_mb:.1f} MB[/] (Parquet)\n"
            f"  Runs:     [cyan]{args.runs}[/] per operation (median reported)\n"
            f"  Engines:  Pandas · Polars · DataFusion · Daft\n"
            f"  CPU:      [cyan]{os.cpu_count()}[/] cores",
            title="⚡ Benchmark Configuration",
            border_style="blue",
        )
    )

    all_results: dict[str, dict] = {}
    tables: dict[str, pa.Table] = {}
    benches = [
        ("Pandas", "red", bench_pandas),
        ("Polars", "blue", bench_polars),
        ("DataFusion", "magenta", bench_datafusion),
        ("Daft", "green", bench_daft),
    ]
    for name, color, bench in benches:
        console.print(f"[bold {color}]▸ Benchmarking {name}...[/]")
        t, tables[name] = bench(trips_glob, zones_str, args.runs)
        all_results[name] = {"ETL Pipeline": t}
        console.print(f"  ✓ {name} done ({t:.3f}s)\n")

    check_results(tables)
    dataset_info["result_rows"] = tables["Pandas"].num_rows
    dataset_info["runs"] = args.runs

    # --- Results ---
    render_table(all_results)
    save_chart(all_results)
    save_json_report(all_results, dataset_info)


if __name__ == "__main__":
    main()
