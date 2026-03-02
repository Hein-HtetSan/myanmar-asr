#!/usr/bin/env python3
"""Fetch all training metrics from MLflow for presentation charts."""
import urllib.request, json, os

def get_history(run_id, metric):
    url = f'http://localhost:5050/api/2.0/mlflow/metrics/get-history?run_id={run_id}&metric_key={metric}'
    try:
        data = json.loads(urllib.request.urlopen(url).read())
        return sorted(data.get('metrics', []), key=lambda x: x['step'])
    except:
        return []

def get_run(run_id):
    url = f'http://localhost:5050/api/2.0/mlflow/runs/get?run_id={run_id}'
    data = json.loads(urllib.request.urlopen(url).read())
    return data['run']

runs = {
    "Whisper Turbo v3": "7d00375de1084438abbc0a8b6379de6e",
    "Dolphin (Whisper-large-v2)": "bf06cef50bff47b0bf7ce26921259f86",
    "SeamlessM4T v2 Large": "d47caa50841f4d9fb82b7ebb7cc91af6",
}

all_data = {}
for name, rid in runs.items():
    print(f"\n=== {name} ===")
    run = get_run(rid)
    status = run['info']['status']
    print(f"  Status: {status}")
    
    metrics_latest = {m['key']: m['value'] for m in run['data']['metrics'] if not m['key'].startswith('system/')}
    for k in sorted(metrics_latest):
        print(f"  {k}: {metrics_latest[k]}")
    
    wer_hist = get_history(rid, 'eval_wer')
    cer_hist = get_history(rid, 'eval_cer')
    loss_hist = get_history(rid, 'loss')
    eval_loss_hist = get_history(rid, 'eval_loss')
    
    all_data[name] = {
        "status": status,
        "run_id": rid,
        "latest": metrics_latest,
        "eval_wer": [(m['step'], m['value']) for m in wer_hist],
        "eval_cer": [(m['step'], m['value']) for m in cer_hist],
        "train_loss": [(m['step'], m['value']) for m in loss_hist],
        "eval_loss": [(m['step'], m['value']) for m in eval_loss_hist],
    }

os.makedirs("viz", exist_ok=True)
with open("viz/mlflow_metrics.json", "w") as f:
    json.dump(all_data, f, indent=2)
print(f"\nSaved to viz/mlflow_metrics.json")
