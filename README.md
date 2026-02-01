# The Engine Wars — Live Demo

**Benchmark Pandas vs. Polars vs. DataFusion vs. Daft on real-world datasets.**

No synthetic data. No toy examples. Real NYC taxi trips and real food photos.

---

## Quick Start

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Sync dependencies
uv sync

# 3. Pre-download datasets (optional — benchmarks auto-download on first run)
uv run python data_loader.py

# 4. Run the tabular benchmark (~2.9M NYC taxi trips)
uv run python bench_tabular.py

# 5. Run the multimodal benchmark (500 real food photos)
uv run python bench_multimodal.py
```

First run downloads ~50 MB of data. Subsequent runs use the cache in `.data/`.

---

## Datasets

### Tabular: NYC Yellow Taxi Trip Records

| Attribute | Value |
|---|---|
| Source | [NYC Taxi & Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| Format | Apache Parquet |
| Default | January 2024 (~2.9M rows × 19 columns, ~45 MB) |
| Join table | Taxi Zone Lookup (265 zones with borough names) |

Real taxi trip records: pickup/dropoff times, locations, distances, fares,
tips, payment types. The join table maps numeric location IDs to human-readable
borough and zone names (e.g., "Manhattan — Upper East Side North").

### Multimodal: Food-101 (ETH Zurich)

| Attribute | Value |
|---|---|
| Source | [ETH Zurich via Hugging Face](https://huggingface.co/datasets/ethz/food101) |
| Format | JPEG images |
| Default | 500 images (configurable) |
| Content | Real food photos — pizza, sushi, steak, etc. |

Real photographs of food in 101 categories. Variable sizes and aspect ratios,
exactly like production ML preprocessing pipelines encounter.

---

## What's Benchmarked

### `bench_tabular.py` — Tabular Operations

| Operation | What it tests | Real-world analogy |
|---|---|---|
| Read Parquet | Full scan of ~45 MB file | Loading a dataset for analysis |
| Filter | `distance > 5mi AND fare > $30` | Finding high-value trips |
| GroupBy + Agg | Revenue by payment type | Payment analytics dashboard |
| Join | Trip data ⟕ Zone lookup | Enriching with borough names |
| ETL Pipeline | Filter → Join → Aggregate → Sort | Building a revenue report |

Engines: **Pandas** · **Polars** · **DataFusion** · **Daft**

### `bench_multimodal.py` — Image Processing

| Operation | What it tests | Real-world analogy |
|---|---|---|
| Load Images | Read + decode JPEGs | ML data pipeline ingestion |
| Resize 224×224 | Resize to model input size | Preprocessing for ResNet/ViT |
| Total Pipeline | Load → Decode → Resize | End-to-end ML preprocessing |

Engines: **Pandas + Pillow** (sequential) vs. **Daft** (parallel Rust)

> Polars and DataFusion are excluded from the multimodal benchmark because
> they lack native image operations — image work would still go through
> sequential Python.

---

## CLI Options

```bash
# Tabular: change data month
uv run python bench_tabular.py --year 2023 --month 6

# Tabular: more timing precision
uv run python bench_tabular.py --runs 5

# Multimodal: more images = larger speedup (more parallelism)
uv run python bench_multimodal.py --images 1000

# Multimodal: quick smoke test
uv run python bench_multimodal.py --images 100
```

---

## Distributed Scripts (Cluster Required)

The `distributed/` directory contains reference implementations for
cluster-scale processing:

| File | Engine | Workload |
|---|---|---|
| `ray_inference.py` | Ray Data | GPU batch image classification |
| `daft_pipeline.py` | Daft Flotilla | Distributed document embedding |
| `spark_etl.py` | PySpark | Petabyte-scale tabular ETL |

Install extras: `uv sync --extra distributed`

These require actual cluster infrastructure (Ray, Spark, or Daft Cloud).

---

## Expected Output

### Tabular benchmark

```
⚡ Engine Wars — NYC Taxi Benchmark Results
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Operation        ┃           Pandas ┃           Polars ┃       DataFusion ┃             Daft ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Read Parquet     │           1.82s  │   0.19s    10×   │   0.15s    12×   │   0.22s     8×   │
│ Filter           │           0.41s  │   0.04s    10×   │   0.03s    14×   │   0.05s     8×   │
│ GroupBy + Agg    │           0.95s  │   0.08s    12×   │   0.07s    14×   │   0.10s    10×   │
│ Join             │           2.10s  │   0.25s     8×   │   0.20s    11×   │   0.30s     7×   │
│ ETL Pipeline     │           3.50s  │   0.30s    12×   │   0.25s    14×   │   0.35s    10×   │
├──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Total            │           8.78s  │   0.86s    10×   │   0.70s    13×   │   1.02s     9×   │
└──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

*(Numbers are illustrative — your results will vary by hardware.)*

Saves `benchmark_results.png` — a grouped bar chart for presentations.

### Multimodal benchmark

Prints a comparison table with wall-clock times and speedup ratios,
plus saves `multimodal_results.png`.

---

## Requirements

- **Python** 3.10 – 3.12
- **uv** (recommended) or pip
- **~2 GB free RAM** for the tabular benchmark
- **Internet** on first run (downloads ~50 MB total, then cached)
- Works on **macOS**, **Linux**, and **Windows**

---

## Project Structure

```
engine-wars-demo/
├── pyproject.toml          # uv/pip project config
├── data_loader.py          # Downloads + caches real datasets
├── bench_tabular.py        # NYC Taxi: Pandas vs Polars vs DataFusion vs Daft
├── bench_multimodal.py     # Food-101: Pandas+Pillow vs Daft (Rust)
├── distributed/
│   ├── ray_inference.py    # Ray Data GPU inference (reference)
│   ├── daft_pipeline.py    # Daft distributed embedding (reference)
│   └── spark_etl.py        # PySpark ETL (reference)
└── .data/                  # Auto-created cache (gitignored)
    ├── nyc_taxi/
    │   ├── yellow_tripdata_2024-01.parquet
    │   └── taxi_zone_lookup.csv
    └── food101/
        ├── food_00000.jpg
        ├── food_00001.jpg
        └── ...
```
