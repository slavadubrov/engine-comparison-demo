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

    // ETL Pipeline: the same job as the Python engines. Read the Parquet files
    // and the zone CSV inside the timed function, filter, join, group, sort,
    // and keep the top 20. The datetime casts align the monthly schemas so
    // the files can be concatenated.
    let job = || {
        let mut dfs = Vec::new();
        for entry in glob(trips_path).expect("Failed to read glob pattern") {
            if let Ok(path) = entry {
                let df = LazyFrame::scan_parquet(path.to_str().unwrap(), Default::default())
                    .unwrap()
                    .with_column(
                        col("tpep_pickup_datetime")
                            .cast(DataType::Datetime(TimeUnit::Microseconds, None)),
                    )
                    .with_column(
                        col("tpep_dropoff_datetime")
                            .cast(DataType::Datetime(TimeUnit::Microseconds, None)),
                    );
                dfs.push(df);
            }
        }

        let trips = concat(dfs, UnionArgs::default()).unwrap();
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
    };
    results.insert("ETL Pipeline".to_string(), timeit(&job, n_runs));

    // Print the result so it can be compared with the Python engines' rows.
    println!("{}", job().head(Some(3)));

    results
}

// ---------------------------------------------------------------------------
// Multimodal Benchmarks (image crate + rayon)
// ---------------------------------------------------------------------------

fn bench_multimodal(images_dir: &str, n_images: usize, n_runs: usize) -> HashMap<String, f64> {
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

    // Same job as the Python engines: read each JPEG, decode to RGB, resize
    // to 224x224 with a bilinear (Triangle) filter. A failed image stops the
    // run, so every engine processes the same images.
    let pipeline_time = timeit(
        || {
            let resized: Vec<_> = image_paths
                .par_iter()
                .map(|p| {
                    let img = image::open(p)
                        .unwrap_or_else(|e| panic!("failed to open {}: {}", p.display(), e))
                        .to_rgb8();
                    image::imageops::resize(&img, 224, 224, FilterType::Triangle)
                })
                .collect();
            assert_eq!(resized.len(), n, "expected one output image per input");
            resized
        },
        n_runs,
    );
    results.insert("Total Pipeline".to_string(), pipeline_time);

    results
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let n_runs = 3;

    // Paths (relative to project root via run script, or run locally)
    let data_dir = "../.data";
    let benchmarks_dir = "../benchmarks";

    // Ensure output directory exists
    fs::create_dir_all(benchmarks_dir).unwrap();

    // --- Tabular Benchmark ---
    // Python saves to .data/nyc_taxi/ with year-month suffix
    let trips_glob = format!("{}/nyc_taxi/yellow_tripdata_2024-*.parquet", data_dir);
    let zones_path = format!("{}/nyc_taxi/taxi_zone_lookup.csv", data_dir);

    // Check if any matching files exist
    let trips_exist = glob(&trips_glob)
        .map(|mut paths| paths.next().is_some())
        .unwrap_or(false);

    if trips_exist && Path::new(&zones_path).exists() {
        println!("🦀 Running Polars-rs tabular benchmark...");

        let tabular_results = bench_tabular(&trips_glob, &zones_path, n_runs);

        // Print results
        println!("\n  Polars-rs Results:");
        for (op, time) in &tabular_results {
            println!("    {}: {:.3}s", op, time);
        }

        // Get row count for dataset info
        let mut rows = 0;
        for entry in glob(&trips_glob).unwrap() {
            if let Ok(path) = entry {
                let df = LazyFrame::scan_parquet(path.to_str().unwrap(), Default::default())
                    .unwrap()
                    .collect()
                    .unwrap();
                rows += df.height();
            }
        }

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
            trips_glob
        );
    }

    // --- Multimodal Benchmark ---
    // Python saves to .data/food101/
    let images_dir = format!("{}/food101", data_dir);

    if Path::new(&images_dir).exists() {
        println!("\n🦀 Running Rust image benchmark...");

        let n_images = 500;
        let multimodal_results = bench_multimodal(&images_dir, n_images, n_runs);

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
