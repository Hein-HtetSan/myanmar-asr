# Research web app

`web/` is the static public research page. It reads `results.json`, generated from MLflow tracking and `metrics/*.json` artifacts. It displays completed runs, numeric result JSON, metric histories, training settings, manifest hashes, and the recursive artifact inventory. The browser has no MLflow credentials. The committed JSON is an empty initial state, not sample accuracy.

## Which runs are exported

The exporter discovers all `FINISHED` runs in `MLFLOW_EXPERIMENT_NAME` (default `myanmar-asr-notebooks`), across all result pages. You do not need to add new run IDs after training.

Full notebook runs are exported only after `metrics/test_metrics.json` exists, since notebook training and test evaluation use separate tracking contexts. Quick runs may be exported once finished with `quick_run: true`. Runs without `configuration/run.json`, failed runs, and running runs are omitted. Every model notebook in this repository writes the expected configuration and test-result artifacts. Legacy runs need the corresponding artifacts before they are exported.

The exporter downloads aggregate JSON artifacts under `metrics/`, reads the allowlisted configuration from `configuration/run.json`, fetches metric histories, and lists all artifacts. It writes `web/results.json` atomically. The site builder then packages only the page and public JSON into `.vercel/output`. Open pages check the published JSON every minute and retain the selected run when it is still present.

There is no CI/CD workflow. Export, build and deploy are run by hand, as described below.

## Run locally

Install `mlflow-skinny==2.19.0` in a separate environment, set the tracker URL, and run from the repository root:

```sh
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
export MLFLOW_EXPERIMENT_NAME=myanmar-asr-notebooks
python scripts/deploy/export_mlflow_results.py
python scripts/deploy/build_research_site.py
python -m http.server 8765 --directory .vercel/output/static
```

To publish the built bundle, disable Vercel's Git auto-deployment for the project, then deploy from the repository root with the Vercel CLI:

```sh
vercel deploy --prebuilt --prod
```

No GPU, Python server, Docker stack, or MLflow credentials run inside the public Vercel page.

For PowerShell, set `$env:MLFLOW_TRACKING_URI = "..."` and `$env:MLFLOW_EXPERIMENT_NAME = "..."`, then run the same Python commands. `--run-ids` can optionally restrict the export; those runs still must be completed and have the required artifacts.

## Published and retained records

MLflow retains full original artifacts. The public export contains run ID/name/status/times, numeric aggregate results, metric histories, allowlisted training configuration and hashes, and artifact paths/sizes. It excludes arbitrary params/tags, environment records, tracker credentials/URLs, local paths, raw transcripts, recordings, logs and weights. Run names and artifact filenames are public; keep sensitive material out of them. No missing score is inferred.

Whisper Small writes `configuration/run.json`, `configuration/environment.json`, `dataset/catalog_summary.json`, `dataset/long_labels.csv`, `metrics/training_metrics.json`, and, after full evaluation, `metrics/test_metrics.json` plus five prediction CSVs under `predictions/`. Other notebooks use equivalent configuration with separate training/validation metric JSONs. Local run directories also keep their own configuration, metrics, checkpoints and saved run ID. Tracking metric histories are records, not automatically generated console-log files.

See [provenance](../docs/REPRODUCIBILITY.md), [citation](../CITATION.cff), [Vercel Build Output API](https://vercel.com/docs/build-output-api/configuration), and [MLflow tracking client](https://mlflow.org/docs/2.19.0/python_api/mlflow.client.html).
