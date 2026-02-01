#!/bin/bash
# Run Spark ETL pipeline on NYC Taxi data
# Usage: ./scripts/docker-run-spark.sh [--orders PATH] [--zones PATH] [--output PATH]

set -e

docker compose exec app bash -c "
    cd /app && \
    spark-submit \
        --master spark://spark-master:7077 \
        --conf spark.hadoop.fs.s3a.endpoint=http://minio:9000 \
        --conf spark.hadoop.fs.s3a.access.key=minioadmin \
        --conf spark.hadoop.fs.s3a.secret.key=minioadmin \
        --conf spark.hadoop.fs.s3a.path.style.access=true \
        --packages org.apache.hadoop:hadoop-aws:3.3.4 \
        src/engine_comparison/distributed/spark_etl.py $@
"
