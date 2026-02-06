#!/usr/bin/env python3
"""
Engine Wars — PySpark: Distributed Tabular ETL on NYC Taxi Data
================================================================
Spark's strength: petabyte-scale joins with fault tolerance and AQE.

This runs the same NYC Taxi ETL pipeline from bench_tabular.py, but
distributed across a Spark cluster for data-lake-scale volumes.

Usage:
  spark-submit spark_etl.py \\
      --orders s3a://lake/nyc_taxi/yellow_tripdata_*.parquet \\
      --zones  s3a://lake/nyc_taxi/taxi_zone_lookup.csv \\
      --output s3a://warehouse/nyc_taxi_report/
"""

from __future__ import annotations

import argparse
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def create_session() -> SparkSession:
    return (
        SparkSession.builder.appName("EngineWars_NYC_Taxi_ETL")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.parquet.filterPushdown", "true")
        .getOrCreate()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders", default="s3a://lake/taxi/*.parquet")
    parser.add_argument("--zones", default="s3a://lake/taxi/taxi_zone_lookup.csv")
    parser.add_argument("--output", default="s3a://warehouse/taxi_report/")
    args = parser.parse_args()

    spark = create_session()
    t0 = time.perf_counter()

    trips = spark.read.parquet(args.orders)
    zones = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(args.zones)
    )

    # Same ETL as bench_tabular: filter → join → aggregate → rank
    window = Window.partitionBy("Borough").orderBy(F.desc("revenue"))

    report = (
        trips.filter((F.col("fare_amount") > 10.0) & (F.col("trip_distance") > 0))
        .join(zones, trips["PULocationID"] == zones["LocationID"], "inner")
        .groupBy("Borough", "Zone")
        .agg(
            F.sum("total_amount").alias("revenue"),
            F.count("*").alias("trips"),
            F.avg("tip_amount").alias("avg_tip"),
            F.avg("trip_distance").alias("avg_distance"),
        )
        .withColumn("rank", F.row_number().over(window))
        .orderBy("Borough", "rank")
    )

    report.cache()
    report.write.partitionBy("Borough").mode("overwrite").parquet(args.output)

    elapsed = time.perf_counter() - t0
    row_count = report.count()

    # --- Summary stats ---
    stats = report.agg(
        F.sum("revenue").alias("total_revenue"),
        F.sum("trips").alias("total_trips"),
        F.countDistinct("Borough").alias("boroughs"),
        F.countDistinct("Zone").alias("zones"),
    ).collect()[0]

    print(f"\n✓ {row_count:,} rows written in {elapsed:.1f}s → {args.output}")
    print(f"  Total revenue: ${stats['total_revenue']:,.2f}")
    print(f"  Total trips:   {stats['total_trips']:,}")
    print(f"  Boroughs: {stats['boroughs']}  |  Zones: {stats['zones']}")

    print("\n── Top 3 Zones per Borough ──")
    report.filter(F.col("rank") <= 3).show(20, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()
