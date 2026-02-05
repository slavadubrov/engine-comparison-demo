//! Rust Native Benchmarks for Engine Comparison Demo
//!
//! Implements tabular (Polars) and multimodal (image crate) benchmarks,
//! outputting JSON results compatible with the Python aggregator.

use chrono::Utc;
use glob::glob;
use image::imageops::FilterType;
use polars::prelude::*;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------------------
// JSON Report Structures
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct BenchmarkReport {
    benchmark: String,
    timestamp: String,
    dataset: serde_json::Value,
    results: HashMap<String, HashMap<String, f64>>,
}

// ---------------------------------------------------------------------------
// Timing Utility
// ---------------------------------------------------------------------------

fn timeit<F, T>(f: F, n_runs: usize) -> f64
where
    F: Fn() -> T,
{
    let mut times: Vec<f64> = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let start = Instant::now();
        let _ = f();
        times.push(start.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2] // median
}

// ---------------------------------------------------------------------------
// Tabular Benchmarks (Polars)
// ---------------------------------------------------------------------------

fn bench_tabular(trips_path: &str, zones_path: &str, n_runs: usize) -> HashMap<String, f64> {
    let mut results = HashMap::new();

    // Read Parquet
    results.insert(
        "Read Parquet".to_string(),
        timeit(
            || {
                LazyFrame::scan_parquet(trips_path, Default::default())
                    .unwrap()
                    .collect()
                    .unwrap()
            },
            n_runs,
        ),
    );

    // Filter: long-distance, high-fare trips
    results.insert(
        "Filter".to_string(),
        timeit(
            || {
                LazyFrame::scan_parquet(trips_path, Default::default())
                    .unwrap()
                    .filter(
                        col("trip_distance")
                            .gt(lit(5.0))
                            .and(col("fare_amount").gt(lit(30.0))),
                    )
                    .collect()
                    .unwrap()
            },
            n_runs,
        ),
    );

    // GroupBy + Agg: revenue by payment type
    results.insert(
        "GroupBy + Agg".to_string(),
        timeit(
            || {
                LazyFrame::scan_parquet(trips_path, Default::default())
                    .unwrap()
                    .group_by([col("payment_type")])
                    .agg([
                        col("VendorID").count().alias("trip_count"),
                        col("total_amount").sum().alias("total_revenue"),
                        col("fare_amount").mean().alias("avg_fare"),
                        col("tip_amount").mean().alias("avg_tip"),
                    ])
                    .collect()
                    .unwrap()
            },
            n_runs,
        ),
    );

    // Join: enrich with pickup zone names
    results.insert(
        "Join".to_string(),
        timeit(
            || {
                let trips = LazyFrame::scan_parquet(trips_path, Default::default()).unwrap();
                let zones = LazyCsvReader::new(zones_path).finish().unwrap();
                trips
                    .join(
                        zones,
                        [col("PULocationID")],
                        [col("LocationID")],
                        JoinArgs::new(JoinType::Inner),
                    )
                    .collect()
                    .unwrap()
            },
            n_runs,
        ),
    );

    // ETL Pipeline: filter → join → groupby → sort → limit
    results.insert(
        "ETL Pipeline".to_string(),
        timeit(
            || {
                let trips = LazyFrame::scan_parquet(trips_path, Default::default()).unwrap();
                let zones = LazyCsvReader::new(zones_path).finish().unwrap();
                trips
                    .filter(col("fare_amount").gt(lit(10.0)))
                    .join(
                        zones,
                        [col("PULocationID")],
                        [col("LocationID")],
                        JoinArgs::new(JoinType::Inner),
                    )
                    .group_by([col("Borough"), col("Zone")])
                    .agg([
                        col("total_amount").sum().alias("revenue"),
                        col("VendorID").count().alias("trips"),
                        col("tip_amount").mean().alias("avg_tip"),
                    ])
                    .sort(
                        ["revenue"],
                        SortMultipleOptions::default().with_order_descending(true),
                    )
                    .limit(20)
                    .collect()
                    .unwrap()
            },
            n_runs,
        ),
    );

    results
}

// ---------------------------------------------------------------------------
// Multimodal Benchmarks (image crate + rayon)
// ---------------------------------------------------------------------------

fn bench_multimodal(images_dir: &str, n_images: usize) -> HashMap<String, f64> {
    let mut results = HashMap::new();

    // Collect image paths
    let pattern = format!("{}/*.jpg", images_dir);
    let image_paths: Vec<_> = glob(&pattern)
        .expect("Failed to read glob pattern")
        .filter_map(|e| e.ok())
        .take(n_images)
        .collect();

    let n = image_paths.len();
    if n == 0 {
        eprintln!("Warning: No images found in {}", images_dir);
        return results;
    }

    // Load Images (parallel) — measure just the load/decode time
    let start = Instant::now();
    let images: Vec<_> = image_paths
        .par_iter()
        .filter_map(|p| image::open(p).ok())
        .collect();
    let load_time = start.elapsed().as_secs_f64();
    results.insert("Load Images".to_string(), load_time);

    // Resize to 224×224 (parallel) — measure just the resize time on pre-loaded images
    let start = Instant::now();
    let _resized: Vec<_> = images
        .into_par_iter()
        .map(|img| img.resize_exact(224, 224, FilterType::Lanczos3))
        .collect();
    let resize_time = start.elapsed().as_secs_f64();
    results.insert("Resize 224×224".to_string(), resize_time);

    // Total Pipeline = Load + Resize (consistent with Python methodology)
    results.insert("Total Pipeline".to_string(), load_time + resize_time);

    results
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let n_runs = 3;

    // Paths (relative to project root)
    let data_dir = ".data";
    let benchmarks_dir = "benchmarks";

    // Ensure output directory exists
    fs::create_dir_all(benchmarks_dir).unwrap();

    // --- Tabular Benchmark ---
    // Python saves to .data/nyc_taxi/ with year-month suffix
    let trips_path = format!("{}/nyc_taxi/yellow_tripdata_2024-01.parquet", data_dir);
    let zones_path = format!("{}/nyc_taxi/taxi_zone_lookup.csv", data_dir);

    if Path::new(&trips_path).exists() && Path::new(&zones_path).exists() {
        println!("🦀 Running Polars-rs tabular benchmark...");

        let tabular_results = bench_tabular(&trips_path, &zones_path, n_runs);

        // Print results
        println!("\n  Polars-rs Results:");
        for (op, time) in &tabular_results {
            println!("    {}: {:.3}s", op, time);
        }

        // Get row count for dataset info
        let df = LazyFrame::scan_parquet(&trips_path, Default::default())
            .unwrap()
            .collect()
            .unwrap();
        let rows = df.height();

        // Save JSON report
        let mut results_map = HashMap::new();
        results_map.insert("Polars-rs".to_string(), tabular_results);

        let report = BenchmarkReport {
            benchmark: "tabular".to_string(),
            timestamp: Utc::now().to_rfc3339(),
            dataset: serde_json::json!({
                "name": "NYC Yellow Taxi",
                "rows": rows,
            }),
            results: results_map,
        };

        let output_path = format!("{}/rust_tabular_results.json", benchmarks_dir);
        let file = File::create(&output_path).unwrap();
        serde_json::to_writer_pretty(BufWriter::new(file), &report).unwrap();
        println!("\n  ✓ JSON saved → {}", output_path);
    } else {
        println!(
            "⚠️  Tabular data not found at {}. Run Python benchmark first to download data.",
            trips_path
        );
    }

    // --- Multimodal Benchmark ---
    // Python saves to .data/food101/
    let images_dir = format!("{}/food101", data_dir);

    if Path::new(&images_dir).exists() {
        println!("\n🦀 Running Rust image benchmark...");

        let n_images = 500;
        let multimodal_results = bench_multimodal(&images_dir, n_images);

        // Print results
        println!("\n  Rust image Results:");
        for (op, time) in &multimodal_results {
            println!("    {}: {:.3}s", op, time);
        }

        // Count actual images
        let pattern = format!("{}/*.jpg", images_dir);
        let actual_count = glob(&pattern)
            .map(|paths| paths.filter_map(|e| e.ok()).count())
            .unwrap_or(0);

        // Save JSON report
        let mut results_map = HashMap::new();
        results_map.insert("Rust image".to_string(), multimodal_results);

        let report = BenchmarkReport {
            benchmark: "multimodal".to_string(),
            timestamp: Utc::now().to_rfc3339(),
            dataset: serde_json::json!({
                "name": "Food-101",
                "images": actual_count.min(n_images),
                "target_size": [224, 224],
            }),
            results: results_map,
        };

        let output_path = format!("{}/rust_multimodal_results.json", benchmarks_dir);
        let file = File::create(&output_path).unwrap();
        serde_json::to_writer_pretty(BufWriter::new(file), &report).unwrap();
        println!("\n  ✓ JSON saved → {}", output_path);
    } else {
        println!(
            "\n⚠️  Image data not found at {}. Run Python benchmark first to download data.",
            images_dir
        );
    }

    println!("\n✅ Rust benchmarks complete!");
}
