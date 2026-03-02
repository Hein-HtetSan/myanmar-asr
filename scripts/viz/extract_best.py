import json

with open("/Users/heinhtetsan/myanmar-asr/viz/mlflow_metrics.json") as f:
    data = json.load(f)

for name, d in data.items():
    wer_hist = d["eval_wer"]
    cer_hist = d["eval_cer"]
    best_wer = min(wer_hist, key=lambda x: x[1])
    best_cer = min(cer_hist, key=lambda x: x[1])
    print(f"=== {name} ===")
    print(f"  Best eval WER: {best_wer[1]}% at step {best_wer[0]}")
    print(f"  Best eval CER: {best_cer[1]}% at step {best_cer[0]}")
    lat = d["latest"]
    print(f"  Latest eval: WER={lat.get('eval_wer')}, CER={lat.get('eval_cer')}")
    if "test_wer" in lat:
        print(f"  Test WER: {lat['test_wer']}%")
        print(f"  Test CER: {lat['test_cer']}%")
    else:
        print(f"  (No separate test metrics logged)")
    rt = lat.get("train_runtime")
    if rt:
        print(f"  Train time: {rt:.0f}s = {rt/60:.1f} min")
    print()
