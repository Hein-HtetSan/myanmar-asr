#!/usr/bin/env python3
"""Quick script to inspect MLflow experiments/runs/artifacts."""
import os
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9002"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin123"

from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5050")
exps = client.search_experiments()
for e in exps:
    print(f"\nExperiment: {e.name} (id={e.experiment_id})")
    runs = client.search_runs([e.experiment_id], max_results=3)
    for r in runs:
        tags = r.data.tags
        name = tags.get("mlflow.runName", r.info.run_id[:8])
        cer = r.data.metrics.get("eval_cer", r.data.metrics.get("test_cer", "N/A"))
        loc = tags.get("model.location", "N/A")
        print(f"  Run: {name} | CER={cer} | model.location={loc}")
        print(f"       artifact_uri={r.info.artifact_uri}")
        try:
            arts = client.list_artifacts(r.info.run_id)
            for a in arts:
                print(f"    artifact: {a.path} (is_dir={a.is_dir}, size={a.file_size})")
                if a.is_dir:
                    sub = client.list_artifacts(r.info.run_id, a.path)
                    for s in sub[:10]:
                        print(f"      {s.path} ({s.file_size})")
        except Exception as ex:
            print(f"    [Error listing artifacts: {ex}]")
