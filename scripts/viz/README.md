# Historical visualization helpers

This directory contains historical experiment metric exports and presentation chart helpers. The existing Python scripts are tracked. Ignore rules exclude additional files in this directory, with an exception for this README; generated visualization artifacts under the repository's top-level `viz/` directory are also ignored.

Run scripts from the repository root. Their data paths are relative to that directory, and their MLflow defaults point to `http://localhost:5050`.

- `fetch_mlflow_data.py`: exports metric histories for hardcoded experiment run IDs into `viz/mlflow_metrics.json`. Requires access to those runs; update the IDs for another experiment.
- `extract_best.py`: reads that JSON and prints minimum validation WER/CER, associated steps, and available test metrics. Histories must contain the expected metric entries.
- `presentation_charts.py`: reads the exported JSON and writes presentation PNGs under `viz/`. Uses NumPy and matplotlib. Some figures include historical constants and manually entered test scores, so they are not fresh evaluations of your current model.
- `backfill_mlflow.py`: historical repair helper that reads a remote log through a hardcoded SSH connection and writes missing metrics to a hardcoded MLflow run. Review the connection, run ID, and step mapping before running; it modifies experiment history.

After configuring the run IDs and making the MLflow server available:

```sh
python scripts/viz/fetch_mlflow_data.py
python scripts/viz/extract_best.py
python scripts/viz/presentation_charts.py
```

For fresh model scores, use [evaluation](../evaluation/README.md). For dataset plots, use [data analysis](../data/README.md). See [result provenance](../../docs/REPRODUCIBILITY.md) before reusing historical charts.
