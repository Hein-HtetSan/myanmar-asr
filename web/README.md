# Public research results

This buildless HTML/CSS/JS page shows selected MLflow runs: latest metrics, complete logged metric histories, an allowlisted training configuration, manifest hashes, and the recursively listed artifact inventory. The browser reads only `results.json`; it never connects to MLflow or receives tracker credentials. The committed snapshot is empty until real runs are exported. No historical or sample accuracy is substituted.

## What MLflow already stores

Whisper Small logs `configuration/run.json`, `configuration/environment.json`, `dataset/catalog_summary.json`, `dataset/long_labels.csv`, `metrics/training_metrics.json`, and, after full evaluation, `metrics/test_metrics.json` and five prediction CSVs under `predictions/`. Training/validation metric histories are MLflow records rather than separate log files. Local `run_config.json`, `training_metrics.json`, `test_metrics.json` and `mlflow_run_id.txt` remain under the run directory.

Other notebooks may use different artifact names. The exporter lists every file recursively, exports all metric histories, and reads `configuration/run.json` when present. Missing configuration is shown explicitly. It does not invent training logs, download every artifact, or imply that listed artifacts are public download links.

## Export a snapshot locally

Use a separate environment with `mlflow-skinny==2.19.0`. Set `MLFLOW_TRACKING_URI` to your existing tracker and `MLFLOW_RUN_IDS` to a comma-separated list of run IDs you have reviewed for public release. On POSIX:

```sh
export MLFLOW_TRACKING_URI=http://127.0.0.1:5050
export MLFLOW_RUN_IDS=YOUR_32_CHARACTER_RUN_ID
python scripts/deploy/export_mlflow_results.py
python scripts/deploy/build_research_site.py
python -m http.server 8765 --directory .vercel/output/static
```

On PowerShell, set the variables using `$env:MLFLOW_TRACKING_URI = "..."` and `$env:MLFLOW_RUN_IDS = "..."`, then run the same Python commands. An export failure preserves the previous JSON and fails the command. Export after test evaluation finishes to include all held-out scores; partially completed/running runs are labeled using their MLflow status. Re-exporting refreshes the snapshot.

Public output includes run ID/name/status/times, numeric metrics, histories, selected training settings and hashes, and artifact filenames/sizes. It excludes arbitrary params/tags, environment details, local paths, tracker URLs, raw artifact contents, transcripts, audio and weights. Review run names and artifact filenames too: they become public. The MLflow service retains the complete originals.

## GitHub Actions and Vercel

`.github/workflows/ci.yml` validates Python/tests, site JavaScript/build, and both Docker Compose files on pushes and pull requests. `.github/workflows/research-results.yml` runs checks on main pushes and manual dispatch, then exports/builds and deploys when enabled. Deployment checks run before secrets are used. Pull requests cannot trigger the publishing workflow.

Add repository Actions **variables**:

- `MLFLOW_EXPORT_ENABLED`: `true` enables the export job.
- `MLFLOW_RUN_IDS`: comma-separated, reviewed run IDs; no automatic publication of every experiment.
- `VERCEL_DEPLOY_ENABLED`: `true` enables the production deployment after a successful export.
- `MLFLOW_RUNNER`: optional runner label; defaults to `ubuntu-latest`.

Add Actions **secrets**:

- `MLFLOW_TRACKING_URI`: tracker address reachable from the export runner.
- `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`, or `MLFLOW_TRACKING_TOKEN`, if your tracker requires them.
- `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`: credentials and IDs for the Vercel project hosting this page. Repository secrets work for the production environment; environment secrets may also be used by the deploy job.

A GitHub-hosted runner cannot reach your laptop's `localhost:5050`. Use an authenticated tracker reachable by the runner, or set `MLFLOW_RUNNER` to a dedicated self-hosted runner label on your trusted network and supply its reachable tracker URL. GitHub runner installation/configuration is external to this repository.

The export job uploads only the public `.vercel/output` bundle as the `web` GitHub artifact, retained for 30 days. The deployment job downloads that exact bundle and runs `vercel deploy --prebuilt --prod`. The Vercel project does not need Python, GPUs, Docker, model weights or access to MLflow. CLI installation currently follows `vercel@latest`; pin a validated version if your deployment policy requires it.

Disable Vercel's separate automatic Git deployments for this project if GitHub Actions owns publication; otherwise its independent build can bypass the selected-run export. Trigger **Publish research results** manually after training to refresh results without changing code. Changing a variable alone does not trigger a deployment. Export/hosting flags default off until configured; missing credentials fail visibly once enabled. No deployment has been executed by this code change.

The Vercel format follows [Build Output API v3](https://vercel.com/docs/build-output-api/configuration) and [prebuilt deployments](https://vercel.com/docs/cli/deploy). MLflow records are read using the [tracking client](https://mlflow.org/docs/2.19.0/python_api/mlflow.client.html). See [research provenance](../docs/REPRODUCIBILITY.md) and [citation](../CITATION.cff).
