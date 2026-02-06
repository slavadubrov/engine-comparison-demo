#!/bin/bash
# Engine Wars — Full Benchmark Pipeline
# ======================================
# Runs Python benchmarks, Rust benchmarks, and aggregates results.
#
# Usage:
#     chmod +x scripts/run_benchmarks.sh
#     ./scripts/run_benchmarks.sh

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         🏁 Engine Wars — Full Benchmark Pipeline             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ---------------------------------------------------------------------------
# Python Benchmarks
# ---------------------------------------------------------------------------

echo "🐍 Running Python tabular benchmark..."
uv run python -m engine_comparison.benchmarks.tabular
echo ""

echo "🐍 Running Python multimodal benchmark..."
uv run python -m engine_comparison.benchmarks.multimodal
echo ""

# ---------------------------------------------------------------------------
# Rust Benchmarks
# ---------------------------------------------------------------------------

echo "🦀 Building and running Rust benchmarks..."
# Ensure cargo is available (rustup installs to ~/.cargo/bin)
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi
# Build in rust_benchmark dir, but run from project root so .data/ paths work
cd rust_benchmark
cargo build --release
cd ..
# Run the binary from project root
./rust_benchmark/target/release/rust_benchmark
echo ""

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

echo "📊 Aggregating results..."
uv run python -m engine_comparison.benchmarks.aggregate
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ✅ All benchmarks complete!               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Output files:"
echo ""

if ls benchmarks/*.json 1> /dev/null 2>&1; then
    echo "   JSON reports:"
    ls -la benchmarks/*.json | awk '{print "     " $NF " (" $5 " bytes)"}'
fi
echo ""

if ls benchmarks/*.png 1> /dev/null 2>&1; then
    echo "   Charts:"
    ls -la benchmarks/*.png | awk '{print "     " $NF}'
fi
echo ""
