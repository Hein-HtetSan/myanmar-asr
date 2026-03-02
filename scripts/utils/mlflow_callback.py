#!/usr/bin/env python3
"""
Enhanced MLflow callback for HuggingFace Trainer.
Logs metrics, params, system/GPU stats, and artifacts to a local MLflow server
reached via SSH reverse tunnel (localhost:5050 on Vast.ai).

Features:
  - Training loss, eval loss, WER, CER per step/epoch
  - GPU utilization, GPU memory, system RAM, CPU usage
  - Model config, hyperparameters, dataset info as params
  - Best checkpoint + training config as artifacts
  - Training curves saved as JSON artifacts for comparison

Usage in training scripts:
    from mlflow_callback import (
        get_mlflow_callback,
        get_system_metrics_callback,
        log_final_results,
        log_model_artifacts,
    )
    callbacks = [
        get_mlflow_callback("whisper-turbo-myanmar"),
        get_system_metrics_callback(),
    ]
    # ... after training:
    log_final_results(test_results, model_name="whisper-turbo")
    log_model_artifacts(output_dir, model_name="whisper-turbo")
"""

import os
import json
import time
import platform
from pathlib import Path


# ── Core Setup ──────────────────────────────────────────

def setup_mlflow(experiment_name="myanmar-asr"):
    """Configure MLflow tracking. Returns True if available."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)
        print(f"  ✓ MLflow tracking: {mlflow_uri} (experiment: {experiment_name})")
        return True
    except Exception as e:
        print(f"  ⚠ MLflow not available: {e}")
        print(f"    Training will continue without MLflow tracking.")
        return False


def get_mlflow_callback(run_name, experiment_name="myanmar-asr"):
    """Get HF Trainer MLflow callback, or None if unavailable."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
    try:
        import mlflow
        from transformers.integrations import MLflowCallback

        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)

        # Set env vars for HF's built-in MLflow integration
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
        os.environ["MLFLOW_RUN_NAME"] = run_name
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "0"  # Don't log full model (too big)

        print(f"  ✓ MLflow callback configured: {mlflow_uri}")
        return MLflowCallback()
    except ImportError:
        print("  ⚠ MLflow not installed — pip install mlflow")
        return None
    except Exception as e:
        print(f"  ⚠ MLflow callback failed: {e}")
        return None


# ── System / GPU Metrics Callback ───────────────────────

def _get_gpu_stats():
    """Get GPU utilization and memory via pynvml or nvidia-smi."""
    stats = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        stats = {
            "gpu/name": name,
            "gpu/utilization_pct": util.gpu,
            "gpu/memory_used_mb": mem.used / 1024**2,
            "gpu/memory_total_mb": mem.total / 1024**2,
            "gpu/memory_pct": (mem.used / mem.total) * 100,
            "gpu/temperature_c": temp,
            "gpu/power_w": power,
        }
        pynvml.nvmlShutdown()
    except Exception:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,nounits,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                vals = result.stdout.strip().split(", ")
                if len(vals) >= 5:
                    stats = {
                        "gpu/utilization_pct": float(vals[0]),
                        "gpu/memory_used_mb": float(vals[1]),
                        "gpu/memory_total_mb": float(vals[2]),
                        "gpu/memory_pct": float(vals[1]) / float(vals[2]) * 100,
                        "gpu/temperature_c": float(vals[3]),
                        "gpu/power_w": float(vals[4]),
                    }
        except Exception:
            pass
    return stats


def _get_system_stats():
    """Get CPU and RAM usage."""
    stats = {}
    try:
        import psutil
        stats["system/cpu_pct"] = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        stats["system/ram_used_mb"] = mem.used / 1024**2
        stats["system/ram_total_mb"] = mem.total / 1024**2
        stats["system/ram_pct"] = mem.percent
    except ImportError:
        pass
    return stats


class SystemMetricsCallback:
    """HF Trainer callback that logs GPU/system metrics every N steps."""

    def __init__(self, log_every_n_steps=10):
        self.log_every_n_steps = log_every_n_steps
        self._start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.time()
        try:
            import mlflow
            # Log system info as params (once)
            mlflow.log_param("system/platform", platform.platform())
            gpu_stats = _get_gpu_stats()
            if "gpu/name" in gpu_stats:
                mlflow.log_param("system/gpu_name", gpu_stats["gpu/name"])
                mlflow.log_param("system/gpu_memory_gb",
                                 f"{gpu_stats.get('gpu/memory_total_mb', 0) / 1024:.1f}")
        except Exception:
            pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return
        try:
            import mlflow
            step = state.global_step

            # GPU metrics
            gpu = _get_gpu_stats()
            for k, v in gpu.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=step)

            # System metrics
            sys_stats = _get_system_stats()
            for k, v in sys_stats.items():
                mlflow.log_metric(k, v, step=step)

            # Training speed
            if self._start_time and step > 0:
                elapsed = time.time() - self._start_time
                mlflow.log_metric("train/steps_per_sec", step / elapsed, step=step)
                mlflow.log_metric("train/elapsed_min", elapsed / 60, step=step)

        except Exception:
            pass  # Never crash training for metrics

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log eval metrics with gpu snapshot."""
        try:
            import mlflow
            step = state.global_step

            # Snapshot GPU during eval
            gpu = _get_gpu_stats()
            for k, v in gpu.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"eval_{k}", v, step=step)

        except Exception:
            pass


def get_system_metrics_callback(log_every_n_steps=10):
    """Get system metrics callback. Uses HF Trainer callback protocol."""
    try:
        from transformers import TrainerCallback

        # Create a proper HF callback wrapper
        class _HFSystemMetrics(TrainerCallback):
            def __init__(self):
                self._inner = SystemMetricsCallback(log_every_n_steps)

            def on_train_begin(self, args, state, control, **kwargs):
                self._inner.on_train_begin(args, state, control, **kwargs)

            def on_log(self, args, state, control, logs=None, **kwargs):
                self._inner.on_log(args, state, control, logs=logs, **kwargs)

            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                self._inner.on_evaluate(args, state, control, metrics=metrics, **kwargs)

        return _HFSystemMetrics()
    except ImportError:
        print("  ⚠ transformers not available for system metrics callback")
        return None


# ── Dataset / Hyperparameter Logging ────────────────────

def log_dataset_info(dataset, model_name=""):
    """Log dataset split sizes and sample info to MLflow."""
    try:
        import mlflow
        if hasattr(dataset, 'num_rows'):
            # Single split
            mlflow.log_param("dataset/total_samples", dataset.num_rows)
        else:
            # DatasetDict
            for split_name in dataset:
                n = len(dataset[split_name])
                mlflow.log_param(f"dataset/{split_name}_samples", n)
            total = sum(len(dataset[s]) for s in dataset)
            mlflow.log_param("dataset/total_samples", total)
    except Exception as e:
        print(f"  ⚠ MLflow dataset info skipped: {e}")


def log_training_config(args_dict, model_config=None, model_name=""):
    """Log training arguments and model config to MLflow."""
    try:
        import mlflow

        # Log key hyperparams
        key_params = [
            "learning_rate", "per_device_train_batch_size", "gradient_accumulation_steps",
            "num_train_epochs", "warmup_steps", "weight_decay", "max_steps",
            "fp16", "bf16", "gradient_checkpointing", "lr_scheduler_type",
        ]
        for k in key_params:
            if k in args_dict:
                mlflow.log_param(f"train/{k}", args_dict[k])

        # Effective batch size
        batch = args_dict.get("per_device_train_batch_size", 1)
        accum = args_dict.get("gradient_accumulation_steps", 1)
        mlflow.log_param("train/effective_batch_size", batch * accum)

        # Model config excerpt
        if model_config:
            config_dict = model_config.to_dict() if hasattr(model_config, 'to_dict') else {}
            for k in ["model_type", "vocab_size", "hidden_size", "num_hidden_layers",
                       "num_attention_heads", "d_model", "encoder_layers", "decoder_layers"]:
                if k in config_dict:
                    mlflow.log_param(f"model/{k}", config_dict[k])

        # Save full config as artifact
        config_path = f"/tmp/{model_name}_training_config.json"
        with open(config_path, "w") as f:
            json.dump(args_dict, f, indent=2, default=str)
        mlflow.log_artifact(config_path)

    except Exception as e:
        print(f"  ⚠ MLflow config logging skipped: {e}")


# ── Final Results + Artifacts ───────────────────────────

def log_final_results(results, model_name, output_dir=None):
    """Log final test results + model info to MLflow."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
    try:
        import mlflow

        mlflow.set_tracking_uri(mlflow_uri)

        # Find active run or start new one
        active_run = mlflow.active_run()
        if active_run is None:
            mlflow.set_experiment("myanmar-asr")
            mlflow.start_run(run_name=f"{model_name}-final")

        # Log test metrics
        for key, value in results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"final/{key}", value)

        # Log summary params
        mlflow.log_param("model_name", model_name)
        if output_dir:
            mlflow.log_param("output_dir", output_dir)

        # Save results as artifact
        results_path = f"/tmp/{model_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        mlflow.log_artifact(results_path)

        if active_run is None:
            mlflow.end_run()

        print(f"  ✓ Final results logged to MLflow")
    except Exception as e:
        print(f"  ⚠ MLflow final logging skipped: {e}")


def log_model_artifacts(output_dir, model_name="model"):
    """Log model checkpoint artifacts (config, tokenizer, results) to MLflow."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)

        active_run = mlflow.active_run()
        if active_run is None:
            mlflow.set_experiment("myanmar-asr")
            mlflow.start_run(run_name=f"{model_name}-artifacts")

        output_path = Path(output_dir)

        # Log key config files (small, informative)
        artifact_files = [
            "config.json", "preprocessor_config.json", "tokenizer_config.json",
            "generation_config.json", "trainer_state.json", "training_args.bin",
            "all_results.json", "train_results.json", "eval_results.json",
        ]
        for fname in artifact_files:
            fpath = output_path / fname
            if fpath.exists():
                mlflow.log_artifact(str(fpath), artifact_path=f"model/{model_name}")

        # Log training curves from trainer_state.json
        trainer_state = output_path / "trainer_state.json"
        if trainer_state.exists():
            with open(trainer_state) as f:
                state = json.load(f)
            log_history = state.get("log_history", [])
            # Extract train/eval metrics over time
            curves = {"train_loss": [], "eval_loss": [], "eval_wer": [], "eval_cer": []}
            for entry in log_history:
                step = entry.get("step", 0)
                if "loss" in entry:
                    curves["train_loss"].append({"step": step, "value": entry["loss"]})
                if "eval_loss" in entry:
                    curves["eval_loss"].append({"step": step, "value": entry["eval_loss"]})
                if "eval_wer" in entry:
                    curves["eval_wer"].append({"step": step, "value": entry["eval_wer"]})
                if "eval_cer" in entry:
                    curves["eval_cer"].append({"step": step, "value": entry["eval_cer"]})

            curves_path = f"/tmp/{model_name}_training_curves.json"
            with open(curves_path, "w") as f:
                json.dump(curves, f, indent=2)
            mlflow.log_artifact(curves_path, artifact_path=f"model/{model_name}")

            # Log best checkpoint info
            best_metric = state.get("best_metric")
            best_checkpoint = state.get("best_model_checkpoint")
            if best_metric is not None:
                mlflow.log_metric("best/metric_value", best_metric)
            if best_checkpoint:
                mlflow.log_param("best/checkpoint", best_checkpoint)

        # Log total model size
        model_files = list(output_path.glob("*.safetensors")) + list(output_path.glob("*.bin"))
        total_size_mb = sum(f.stat().st_size for f in model_files) / 1024**2
        if total_size_mb > 0:
            mlflow.log_metric("model/size_mb", total_size_mb)

        if active_run is None:
            mlflow.end_run()

        print(f"  ✓ Model artifacts logged to MLflow ({len(artifact_files)} files checked)")
    except Exception as e:
        print(f"  ⚠ MLflow artifact logging skipped: {e}")


# ── Comparison Helper ───────────────────────────────────

def get_comparison_table(experiment_name="myanmar-asr"):
    """Fetch all runs and return a comparison dict for display."""
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5050")
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return {}

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.`final/eval_wer` ASC"],
        )
        return runs.to_dict() if hasattr(runs, 'to_dict') else {}
    except Exception as e:
        print(f"  ⚠ Comparison fetch failed: {e}")
        return {}

