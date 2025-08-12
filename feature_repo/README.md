Setup Feast (local)

1) Install

```
pip install feast pyarrow
```

2) Initialize registry (from project root)

```
cd feature_repo
feast apply
```

3) Materialize offline -> online (optional)

```
feast materialize-incremental $(python -c "import datetime; print(datetime.datetime.utcnow().isoformat()+'Z')")
```

Notes
- The FileSource points to `Data/feature_store/karachi_daily_features.parquet` produced by `Data_Collection/feature_store_pipeline.py`.
- Ensure you have generated the Parquet by running the pipeline first.

