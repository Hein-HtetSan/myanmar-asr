#!/usr/bin/env python3
"""Quick evaluation of trained Whisper model on test set."""
import torch, json, os
from datasets import load_from_disk, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import evaluate

MODEL_DIR = "/workspace/models/whisper-turbo-myanmar"
DATA_DIR  = "/workspace/data/myanmar_asr"

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 7 else torch.float32
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_DIR, torch_dtype=dtype
).to(device)
model.eval()

print("Loading test set...")
ds = load_from_disk(DATA_DIR).cast_column("audio", Audio(sampling_rate=16000))
test_ds = ds["test"]
print(f"  Test samples: {len(test_ds)}")

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

all_preds = []
all_refs  = []

print("Evaluating...")
batch_size = 1 if device == "cuda" and dtype == torch.float32 else 8
for i in range(0, len(test_ds), batch_size):
    batch = test_ds[i:min(i+batch_size, len(test_ds))]
    inputs = processor(
        [audio["array"] for audio in batch["audio"]], sampling_rate=16000,
        return_tensors="pt", padding=True
    )
    input_features = inputs.input_features.to(device, dtype=dtype)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_features,
            language="my",
            task="transcribe",
            max_new_tokens=225,
        )
    
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
    refs  = batch["sentence"]
    all_preds.extend(preds)
    all_refs.extend(refs)
    
    if (i // batch_size) % 20 == 0:
        print(f"  Processed {min(i+batch_size, len(test_ds))}/{len(test_ds)}")

wer = 100 * wer_metric.compute(predictions=all_preds, references=all_refs)
cer = 100 * cer_metric.compute(predictions=all_preds, references=all_refs)
print(f"\n{'='*40}")
print(f"Test Results (28h model)")
print(f"{'='*40}")
print(f"  WER: {wer:.2f}%")
print(f"  CER: {cer:.2f}%")
print(f"  Samples: {len(all_refs)}")

# Save results
results = {"test_wer": round(wer, 2), "test_cer": round(cer, 2), "samples": len(all_refs)}
os.makedirs("/workspace/results", exist_ok=True)
with open("/workspace/results/test_results_28h.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to /workspace/results/test_results_28h.json")

# Show sample predictions
print(f"\n{'='*40}")
print("Sample Predictions")
print(f"{'='*40}")
for i in range(min(5, len(all_preds))):
    print(f"  REF: {all_refs[i][:100]}")
    print(f"  HYP: {all_preds[i][:100]}")
    print()
