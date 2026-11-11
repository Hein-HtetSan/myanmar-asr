# Optional local services

Whisper Small and the larger-model notebooks use MLflow. Reuse your running tracker by setting `MLFLOW_TRACKING_URI` before launching Jupyter.

If a tracker is not already running, start the dedicated notebook service from the repository root:

```sh
docker compose -f services/mlflow-notebooks.compose.yml up -d
```

Open http://127.0.0.1:5050. See [tracking instructions](../notebook/README.md) for port configuration, persistent storage and remote kernels.

For the full annotation/storage stack, copy `.env.example` to `.env`, set unique credentials, and run `docker compose -f services/docker-compose.yml up -d`. Keep this stack on a trusted local network. The two stacks share the default MLflow port; run one or configure a separate port. Back up volumes separately. `down` preserves them; `down -v` deletes them.
