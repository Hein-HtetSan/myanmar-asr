# Myanmar ASR Whisper Small training notebook

[`train.ipynb`](train.ipynb) fine-tunes `openai/whisper-small` for Myanmar transcription using every source in the Whisper manifests under `notebook/dataset-mapping/`. It trains on `train_whisper.csv`, selects checkpoints with `validation_whisper.csv`, and evaluates `test_asr_whisper.csv`, `test_user_study_whisper.csv`, and `test_ucsy_whisper.csv` separately. Audio paths in those CSVs are relative to `dataset/`. The audio and generated manifests are not included in this repository. See the [dataset mapping notes](../dataset-mapping/README.md) for split definitions.

## Set up on Windows

1. Start PowerShell in the repository root, with the dataset and manifests in the paths above. Activate the Python environment that Jupyter will use.
2. Install PyTorch for your GPU in that environment using the [official PyTorch selector](https://pytorch.org/get-started/locally/). Install Jupyter if needed: `python -m pip install jupyterlab ipykernel`.
3. Make sure `ffmpeg` and `ffprobe` work from PowerShell. Run `jupyter lab` from inside the repository and open `notebook/whisper-small/train.ipynb`.
4. Run cells from the top. The first code cell installs the remaining notebook packages, including Transformers 4.57.1. Restart the kernel if installation asks you to, then rerun from the top.

The hardware cell should print `device: cuda` and your GPU name. `Current GPU: 0` means PyTorch selected the first GPU; it does not measure GPU activity. In another PowerShell window, run `nvidia-smi -l 1` to watch utilization, memory, temperature, and power while training.

## Choose a run

`QUICK_RUN = False` is the default. It uses every clip in the training and validation Whisper manifests that fits Whisper's target-token limit, trains for three epochs, and evaluates all three held-out test manifests after training. The notebook prints the count for each source and split. The `*_whisper.csv` manifests already exclude audio longer than 30 seconds; the notebook reports how many labels exceed the model's token limit.

For a short pipeline check, set `QUICK_RUN = True`, restart the kernel, and rerun all cells. It selects up to 256 training and 64 validation clips, trains for 20 optimizer steps, and skips the test sets. Quick and full runs use separate output folders:

| Mode | Output folder | Checkpoint interval |
| --- | --- | ---: |
| Quick | `notebook/whisper-small/run_all_quick/` | 5 optimizer steps |
| Full | `notebook/whisper-small/run_all_full/` | 50 optimizer steps |

Training uses a GPU batch size of 1 with 16 gradient accumulation steps. Evaluation uses a batch size of 1. Up to three checkpoints are retained, including the best evaluated checkpoint. Each checkpoint stores model, optimizer, scheduler, and random state so training can resume; keep enough free disk space for several large checkpoints.

## Resume after a power cut

Keep the output folder on disk. After power returns, reopen Jupyter, keep the same `QUICK_RUN` setting, and run the notebook from the top. The training cell reuses saved baseline metrics and resumes automatically from the newest checkpoint with the expected files and a valid trainer state. If power was lost during a save, it tries an older checkpoint. Work since the last completed checkpoint must be repeated. The new all-source folders keep these checkpoints separate from earlier UCSY-only runs.

Keep the same dataset, model, and training settings when resuming. For a new experiment, use a new output folder. Checkpoints are local files and are ignored by Git; copying this repository alone does not back them up.

After training, the selected model is saved in `best_model/` under the mode's output folder. Full-run metrics and predictions are written to `run_all_full/`, with separate files for `test_asr`, `test_user_study`, and `test_ucsy`.

Checkpointing limits lost work from an outage. It does not fix a CUDA error or a monitor that stays black during training; check the Windows System event log for GPU driver errors before retrying that failure.
