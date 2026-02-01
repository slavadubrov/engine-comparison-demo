"""
Constants for engine-comparison-demo benchmarks.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Image Processing
# ---------------------------------------------------------------------------

DEFAULT_N_IMAGES = 5000
"""Default number of Food-101 images to download/benchmark."""

TARGET_IMAGE_SIZE = (224, 224)
"""Standard vision model input size (ResNet, ViT, etc.)."""

# ---------------------------------------------------------------------------
# Taxi Data
# ---------------------------------------------------------------------------

DEFAULT_TAXI_YEAR = 2024
"""Default year for NYC Taxi data."""

DEFAULT_TAXI_MONTH = 1
"""Default month for NYC Taxi data."""

TAXI_TRIPS_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_{year}-{month:02d}.parquet"
)
"""URL template for NYC Yellow Taxi trip data (Parquet)."""

TAXI_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
"""URL for NYC Taxi zone lookup CSV."""

# ---------------------------------------------------------------------------
# Benchmark Settings
# ---------------------------------------------------------------------------

DEFAULT_BENCHMARK_RUNS = 3
"""Default number of timing runs per operation (median reported)."""

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(".data")
"""Directory for cached downloaded data."""

BENCHMARKS_OUTPUT_DIR = Path("benchmarks")
"""Directory for benchmark output charts."""

TABULAR_CHART_OUTPUT = BENCHMARKS_OUTPUT_DIR / "benchmark_results.png"
"""Output path for tabular benchmark chart."""

MULTIMODAL_CHART_OUTPUT = BENCHMARKS_OUTPUT_DIR / "multimodal_results.png"
"""Output path for multimodal benchmark chart."""
