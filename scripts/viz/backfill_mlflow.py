#!/usr/bin/env python3
"""Backfill missing Seamless metrics from remote log into MLflow."""
import json, requests, subprocess, re, sys

RUN_ID = "d47caa50841f4d9fb82b7ebb7cc91af6"
MLFLOW = "http://localhost:5050"

# Fetch log from remote
print("Fetching log from remote...")
result = subprocess.run(
    ["ssh", "-p", "37960", "-i", "/Users/heinhtetsan/.ssh/id_ed25519",
     "root@ssh6.vast.ai", "cat /workspace/logs/train_seamless_v2.log"],
    capture_output=True, text=True, timeout=30
)
log = result.stdout
print(f"Log size: {len(log)} chars")

# Parse training entries
train_pattern = r"\{'loss': '([0-9.]+)', 'grad_norm': '([0-9.]+)', 'learning_rate': '([0-9e.\-]+)', 'epoch': '([0-9.]+)'\}"
train_matches = re.findall(train_pattern, log)
print(f"Training entries in log: {len(train_matches)}")

# Parse eval entries
eval_pattern = r"\{'eval_loss': '([0-9.]+)', 'eval_wer': '([0-9.]+)', 'eval_cer': '([0-9.]+)'.+?'epoch': '([0-9.]+)'\}"
eval_matches = re.findall(eval_pattern, log)
print(f"Eval entries in log: {len(eval_matches)}")

# Get existing MLflow data
resp = requests.get(f"{MLFLOW}/api/2.0/mlflow/metrics/get-history",
                    params={"run_id": RUN_ID, "metric_key": "loss", "max_results": 5000})
existing_loss = set(m["step"] for m in resp.json()["metrics"])
print(f"Existing MLflow loss points: {len(existing_loss)}")

resp = requests.get(f"{MLFLOW}/api/2.0/mlflow/metrics/get-history",
                    params={"run_id": RUN_ID, "metric_key": "eval_wer", "max_results": 500})
existing_eval = set(m["step"] for m in resp.json()["metrics"])
print(f"Existing MLflow eval points: {len(existing_eval)}")

# Build log data with step mapping
log_metrics = {}
for i, (loss, gn, lr, ep) in enumerate(train_matches):
    step = 190 + i * 10
    log_metrics[step] = {
        "loss": float(loss), "grad_norm": float(gn),
        "learning_rate": float(lr), "epoch": float(ep),
    }

# Find and backfill missing training steps
missing_train = sorted(set(log_metrics.keys()) - existing_loss)
print(f"\nMissing training steps: {len(missing_train)}")

if missing_train:
    print(f"  Steps: {missing_train}")
    metrics = []
    for step in missing_train:
        m = log_metrics[step]
        for key in ["loss", "grad_norm", "learning_rate", "epoch"]:
            metrics.append({"key": key, "value": m[key], "step": step, "timestamp": 0})
    r = requests.post(f"{MLFLOW}/api/2.0/mlflow/runs/log-batch",
                      json={"run_id": RUN_ID, "metrics": metrics})
    print(f"  Backfilled: HTTP {r.status_code}")

# Find and backfill missing eval steps
eval_step_list = [364 + i * 182 for i in range(len(eval_matches))]
missing_eval_data = []
for step, (el, ew, ec, ep) in zip(eval_step_list, eval_matches):
    if step not in existing_eval:
        missing_eval_data.append((step, float(el), float(ew), float(ec)))

print(f"Missing eval steps: {len(missing_eval_data)}")
if missing_eval_data:
    for step, el, ew, ec in missing_eval_data:
        print(f"  step={step}: WER={ew}, CER={ec}, loss={el}")
    metrics = []
    for step, el, ew, ec in missing_eval_data:
        metrics.extend([
            {"key": "eval_loss", "value": el, "step": step, "timestamp": 0},
            {"key": "eval_wer", "value": ew, "step": step, "timestamp": 0},
            {"key": "eval_cer", "value": ec, "step": step, "timestamp": 0},
        ])
    r = requests.post(f"{MLFLOW}/api/2.0/mlflow/runs/log-batch",
                      json={"run_id": RUN_ID, "metrics": metrics})
    print(f"  Backfilled: HTTP {r.status_code}")

# Verify
resp = requests.get(f"{MLFLOW}/api/2.0/mlflow/metrics/get-history",
                    params={"run_id": RUN_ID, "metric_key": "loss", "max_results": 5000})
final_loss = len(resp.json()["metrics"])
resp = requests.get(f"{MLFLOW}/api/2.0/mlflow/metrics/get-history",
                    params={"run_id": RUN_ID, "metric_key": "eval_wer", "max_results": 500})
final_eval = len(resp.json()["metrics"])
print(f"\nFinal: {final_loss} loss points, {final_eval} eval points")
print("Done!")
