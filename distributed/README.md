# Distributed Benchmark Scripts

Reference implementations for cluster-scale processing. These require actual
infrastructure (Ray cluster, Spark cluster, or Daft Cloud) to run.

```bash
uv sync --extra distributed
```

| Script | Engine | Workload | Cluster |
|---|---|---|---|
| `ray_inference.py` | Ray Data | GPU batch image classification | Ray + GPU nodes |
| `daft_pipeline.py` | Daft Flotilla | Distributed document embedding | Ray / Daft Cloud |
| `spark_etl.py` | PySpark | Multi-table ETL with window functions | Spark (Databricks, EMR) |
