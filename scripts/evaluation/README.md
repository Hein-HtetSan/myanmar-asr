# Evaluation and dataset analysis

Run commands from the repository root in an environment with PyTorch, Transformers, datasets, evaluate, jiwer, NumPy, and pandas. Plotting also uses matplotlib and seaborn. These scripts use configuration constants rather than command-line path arguments; review them before running.

## Scripts

- `quick_eval.py`: evaluates one Whisper model on the saved dataset's `test` split. Set `MODEL_DIR` to a directory containing model and processor files, such as a trainer's `final/` directory. Set `DATA_DIR` to your Hugging Face dataset. Defaults target `/workspace/models/whisper-turbo-myanmar` and `/workspace/data/myanmar_asr`. Prints WER/CER percentages and sample predictions; writes `/workspace/results/test_results_28h.json`. The historical filename does not measure the dataset's current duration.
- `evaluate_models.py`: compares configured Whisper and Seamless model paths or Hub IDs on the same test split, including available Canary results from a separate JSON file. Configure `DATASET_PATH`, model dictionaries, `CANARY_RESULTS`, and `RESULTS_DIR`. Writes `model_comparison.csv` and `model_comparison.json` under `/workspace/results/` by default, and attempts tracking integration. Its local paths may differ from current trainer defaults.
- `analyze_dataset.py`: reads `/workspace/data/myanmar_asr` and writes dataset plots and a JSON summary under `/workspace/analysis/`. For an analysis script with a positional local dataset path, use [data/analyze_dataset.py](../data/analyze_dataset.py).

After configuring the paths:

```sh
python scripts/evaluation/quick_eval.py
python scripts/evaluation/evaluate_models.py
```

Both evaluators expect a Hugging Face `DatasetDict` with `audio` and `sentence` columns, not the Whisper Small notebook's CSV manifests. Evaluation needs usable model weights and processor files; it does not restore training optimizer state.

Use a held-out test split after selecting the model on validation data. The evaluation scripts report WER and CER as percentages using their supplied reference and prediction text; they do not apply the notebook's `no_spaces` scoring function. Myanmar spacing differences can therefore change comparisons. Keep the exact model, splits, and scoring policy with any reported results. See [reproducibility](../../docs/REPRODUCIBILITY.md).
