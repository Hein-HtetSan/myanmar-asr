# Myanmar ASR

**Burmese speech recognition research toolkit** · [MIT code license](LICENSE) · [Cite this project](CITATION.cff)

This repository documents a comparative Myanmar ASR study and provides data preparation, fine-tuning, evaluation, and inference code. It compares two Whisper checkpoints and SeamlessM4T v2. The project also includes a compatibility path for a CentOS 7 VPS with CUDA 11.4 and a Tesla K80. Historical results, visualizations, and the slide deck are kept in local versioned archives outside this repository.

## Research results

| Model | Pretrained checkpoint | Reported test WER ↓ | Reported test CER ↓ | Training time |
| --- | --- | ---: | ---: | ---: |
| Whisper v3 Turbo | [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo) | 53.50% | 34.78% | 159 min |
| “Dolphin” (fine-tuned Whisper v2) | [`openai/whisper-large-v2`](https://huggingface.co/openai/whisper-large-v2) | **33.02%** | 28.00% | 335 min |
| SeamlessM4T v2 Large | [`facebook/seamless-m4t-v2-large`](https://huggingface.co/facebook/seamless-m4t-v2-large) | 49.12% | **13.04%** | 239 min |

The test values were entered into [`scripts/viz/presentation_charts.py`](scripts/viz/presentation_charts.py) for the thesis figures; the repository does not include test predictions or the final dataset needed to regenerate them. Exported MLflow metrics, analysis figures, and historical reports are stored locally outside this repository. The best WER and CER belong to different models. WER is sensitive to Burmese word spacing, so both metrics and the text normalization policy matter. See [reproducibility and evidence](docs/REPRODUCIBILITY.md) before comparing or reusing the numbers.

The original runs used an RTX 4090 with PyTorch 2.5.1, CUDA 12.1, and Transformers 5.2.0. These timings are **not** K80 estimates. The archived training progress report records experiment decisions and intermediate results.

## What this project includes

- Dataset scripts for downloading, filtering, normalizing, augmenting, and exporting Burmese speech from FLEURS, OpenSLR 80, and YODAS.
- Fine-tuning scripts for Whisper Turbo, Whisper v2, SeamlessM4T v2, and an experimental NeMo Canary path.
- WER/CER evaluation, MLflow metric export and figure-generation scripts, a FastAPI inference service, and a Streamlit demo.
- A [CentOS 7 / K80 decision and VPS test runbook](docs/CentOS7_K80_Compatibility_Decision.md).

The source datasets have their own licenses: [FLEURS](https://huggingface.co/datasets/google/fleurs) is CC BY 4.0, [OpenSLR 80](https://openslr.org/80/) is CC BY-SA 4.0, and [YODAS](https://huggingface.co/datasets/espnet/yodas) is CC BY 3.0. Meta's [SeamlessM4T v2 checkpoint](https://huggingface.co/facebook/seamless-m4t-v2-large) is labeled CC BY-NC 4.0. The repository does not redistribute recordings or model weights. See the [data and result provenance note](docs/REPRODUCIBILITY.md).

## Optional local services

To run the MLflow, MinIO, Label Studio, Argilla, PostgreSQL, and Redis stack, copy `.env.example` to `.env`, fill in unique credentials, then run `docker compose -f services/docker-compose.yml up -d`. Keep this stack on a trusted local network; its service ports are not configured for public exposure. The slide deck and its Docker bundle are stored locally outside this repository.

## Run inference on a CentOS 7 K80 VPS

Use the [VPS runbook](docs/CentOS7_K80_Compatibility_Decision.md#vps-deployment-runbook) for installation, a systemd service template, API requests, and a pass/fail test record. The short version is:

```bash
conda create -n myanmar-asr-k80 python=3.10 -y
conda activate myanmar-asr-k80
python -m pip install 'torch==1.12.1+cu113' --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install -r requirements-k80.txt
python -m pip check
python scripts/infra/check_k80.py --model openai/whisper-tiny
SERVE_HOST=127.0.0.1 DEFAULT_MODEL=whisper-small python scripts/deploy/inference_server.py
```

The K80 uses float32. The smoke test and a real audio request must pass on the target VPS before calling the deployment verified. For a fine-tuned Whisper Small checkpoint, set `WHISPER_SMALL_PATH` to its `final/` directory. The original large-model training runs are not validated on one 12 GB K80 GPU.

## Reproduce or extend the study

The data scripts expect external audio and generate Hugging Face `DatasetDict` splits with `audio` and `sentence` columns. Supply that dataset and explicit model/output paths to a trainer. For a feasible K80 starting run:

```bash
python scripts/training/train_whisper_turbo.py \
  --model_id openai/whisper-small --batch_size 1 --grad_accum 32 \
  --freeze_encoder --no_push --dataset_path /path/to/myanmar_asr \
  --output_dir ./models/whisper-small-k80
```

The script name comes from its original Turbo experiment; `--model_id` selects Whisper Small here. Preserve split definitions, preprocessing settings, checkpoint revision, and prediction outputs for any new comparison. The locally archived dataset statistics JSON describes an earlier 12,299-example, 24.34-hour snapshot; the thesis presentation reports a later 22,705-example, 54.2-hour corpus. The later corpus manifest is not committed. See [reproducibility](docs/REPRODUCIBILITY.md) for the distinction.

## Repository map

| Path | Contents |
| --- | --- |
| [`scripts/data/`](scripts/data) | Data collection and preparation |
| [`scripts/training/`](scripts/training) | Model fine-tuning experiments |
| [`scripts/evaluation/`](scripts/evaluation) | Evaluation and dataset analysis |
| [`scripts/deploy/`](scripts/deploy) | FastAPI inference and Streamlit demo |
| [`scripts/infra/`](scripts/infra) | GPU checks, annotation, and Vast.ai helpers |
| [`scripts/viz/`](scripts/viz) | Metric export and figure-generation code |
| [`docs/`](docs) | Reproducibility and VPS runbooks |

## Contributing, citation, and license

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Cite this software with [CITATION.cff](CITATION.cff), and cite any upstream datasets or models used in your work. Original code and documentation are available under the [MIT license](LICENSE). Dataset and pretrained-model licenses remain with their respective owners.
