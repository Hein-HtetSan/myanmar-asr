# Research web app

`web/` is the static public research page. It reads `results.json`, generated from MLflow tracking and `metrics/*.json` artifacts. It displays completed runs, numeric result JSON, metric histories, training settings, manifest hashes, and the recursive artifact inventory. The browser has no MLflow credentials. The committed JSON is an empty initial state, not sample accuracy.

## Automatic updates after experiments

`.github/workflows/research-results.yml` checks MLflow every five minutes, on main pushes, and on manual dispatch. It discovers all `FINISHED` runs in `MLFLOW_EXPERIMENT_NAME` (default `myanmar-asr-notebooks`), across all result pages. You do not need to add new run IDs after training.

Full notebook runs are published only after `metrics/test_metrics.json` exists, since notebook training and test evaluation use separate tracking contexts. Quick runs may publish once finished with `quick_run: true`. Runs without `configuration/run.json`, failed runs, and running runs are omitted. Every model notebook in this repository writes the expected configuration and test-result artifacts. Legacy runs need the corresponding artifacts before automatic publication.

The exporter downloads aggregate JSON artifacts under `metrics/`, reads the allowlisted configuration from `configuration/run.json`, fetches metric histories, and lists all artifacts. It writes `web/results.json` atomically, then packages only the page and public JSON into `.vercel/output`. The deploy job publishes that bundle to Vercel. A content fingerprint skips redeployment when the page and completed-run data are unchanged; export timestamps alone do not trigger a new deployment. A cache entry is saved only after successful deployment.

Updates are automatic snapshots rather than instant webhooks. GitHub can delay scheduled jobs, and scheduling starts after the workflow is on the repository's default branch. Open pages check the published JSON every minute and retain the selected run when it is still present. Manually dispatch **Publish research results** to request an earlier refresh. The previous deployment stays available if export or deployment fails.

## Configure GitHub Actions

Set repository Actions **variables**:

- `MLFLOW_EXPORT_ENABLED=true`: enable automatic discovery/export.
- `VERCEL_DEPLOY_ENABLED=true`: enable production deployment after checks/export succeed.
- `MLFLOW_EXPERIMENT_NAME`: optional experiment container name; defaults to `myanmar-asr-notebooks`.
- `MLFLOW_RUN_IDS`: optional comma-separated run allowlist; leave unset to include every completed eligible run in the configured experiment.
- `MLFLOW_RUNNER`: optional dedicated runner label; defaults to `ubuntu-latest`.

Set Actions **secrets**:

- `MLFLOW_TRACKING_URI`: existing tracker URL reachable from the export runner.
- `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`, or `MLFLOW_TRACKING_TOKEN`, when required.
- `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`: target hosting project credentials and IDs.

The deploy job uses the `production` environment. Repository secrets work; production environment secrets may be used for Vercel too. GitHub-hosted runners cannot reach your laptop's `localhost:5050`. Use a tracker reachable from the runner or a dedicated self-hosted runner on your trusted network with an appropriate tracker URL. The export flags default off until configured; missing credentials then fail visibly.

Disable Vercel's independent Git auto-deployment for this project so it cannot bypass the MLflow export. The workflow uploads the public bundle as the `web` Actions artifact for 30 days and deploys it with `vercel deploy --prebuilt --prod`. No GPU, Python server, Docker stack, or MLflow credentials run inside the public Vercel page. The CLI currently uses `vercel@latest`; pin a verified version if required by your policy. If you change the Vercel target project, delete the `research-published-*` Actions cache entries or change the page to force a fresh deployment.

## Run locally

Install `mlflow-skinny==2.19.0` in a separate environment, set the tracker URL, and run from the repository root:

```sh
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
export MLFLOW_EXPERIMENT_NAME=myanmar-asr-notebooks
python scripts/deploy/export_mlflow_results.py
python scripts/deploy/build_research_site.py
python -m http.server 8765 --directory .vercel/output/static
```

For PowerShell, set `$env:MLFLOW_TRACKING_URI = "..."` and `$env:MLFLOW_EXPERIMENT_NAME = "..."`, then run the same Python commands. `--run-ids` can optionally restrict the export; those runs still must be completed and have the required artifacts.

## Published and retained records

MLflow retains full original artifacts. The public export contains run ID/name/status/times, numeric aggregate results, metric histories, allowlisted training configuration and hashes, and artifact paths/sizes. It excludes arbitrary params/tags, environment records, tracker credentials/URLs, local paths, raw transcripts, recordings, logs and weights. Run names and artifact filenames are public; keep sensitive material out of them. No missing score is inferred.

Whisper Small writes `configuration/run.json`, `configuration/environment.json`, `dataset/catalog_summary.json`, `dataset/long_labels.csv`, `metrics/training_metrics.json`, and, after full evaluation, `metrics/test_metrics.json` plus five prediction CSVs under `predictions/`. Other notebooks use equivalent configuration with separate training/validation metric JSONs. Local run directories also keep their own configuration, metrics, checkpoints and saved run ID. Tracking metric histories are records, not automatically generated console-log files.

See [provenance](../docs/REPRODUCIBILITY.md), [citation](../CITATION.cff), [Vercel Build Output API](https://vercel.com/docs/build-output-api/configuration), [MLflow tracking client](https://mlflow.org/docs/2.19.0/python_api/mlflow.client.html), and [GitHub scheduled workflow behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onschedule).
