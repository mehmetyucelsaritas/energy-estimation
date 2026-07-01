#!/usr/bin/env python3
"""End-to-end LitePred training on a connected Android CPU device (Pixel 9)."""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
LITEPRED_ROOT = REPO_ROOT / "LitePred"
sys.path.insert(0, str(REPO_ROOT))
import importlib.util


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_profile_dir = LITEPRED_ROOT / "profile_script"
_bench_utils = _load_module("litepred_bench_utils", _profile_dir / "bench_utils.py")
sys.modules["bench_utils"] = _bench_utils
_run_on_device = _load_module("litepred_run_on_device", _profile_dir / "run_on_device.py")
_adb_connect = _load_module("litepred_adb_connect", _profile_dir / "ADBConnect.py")

ADBConnect = _adb_connect.ADBConnect
run_on_android = _run_on_device.run_on_android

sys.path.insert(0, str(LITEPRED_ROOT / "predictor_builder"))

from nn_meter.builder.kernel_predictor_builder.data_sampler.utils import (  # noqa: E402
    generate_model_for_kernel,
    get_sampler_for_kernel,
)
from nn_meter.builder.kernel_predictor_builder.predictor_builder.utils import (  # noqa: E402
    get_flops_params,
)
from nn_meter.utils import latency_metrics  # noqa: E402
from detector import Detector  # noqa: E402
from trainer import Trainer  # noqa: E402

KERNELS = ["conv-bn-relu", "fc", "avgpool", "maxpool"]

DEFAULT_SAMPLE_COUNTS = {
    "conv-bn-relu": 350,
    "fc": 200,
    "avgpool": 100,
    "maxpool": 100,
}

FEATURE_COLUMNS = {
    "conv-bn-relu": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES", "FLOPS", "PARAMS"],
    "dwconv-bn-relu": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES", "FLOPS", "PARAMS"],
    "fc": ["CIN", "COUT", "FLOPS", "PARAMS"],
    "avgpool": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
    "maxpool": ["HW", "CIN", "COUT", "KERNEL_SIZE", "STRIDES"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train LitePred predictors on Pixel 9 CPU.")
    parser.add_argument("--device-serial", default=None, help="ADB device serial (auto-detect if omitted).")
    parser.add_argument(
        "--output-root",
        default=str(LITEPRED_ROOT / "outputs"),
        help="Directory for run artifacts.",
    )
    parser.add_argument(
        "--predictor-pool",
        default=str(LITEPRED_ROOT / "predictors" / "pool"),
        help="LitePred transfer-learning predictor pool.",
    )
    parser.add_argument("--benchmark-path", default="/data/local/tmp/benchmark_model")
    parser.add_argument("--remote-model-dir", default="/sdcard")
    parser.add_argument("--taskset", default="60", help="CPU affinity mask for on-device benchmarking.")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--warm-ups", type=int, default=30)
    parser.add_argument("--num-runs", type=int, default=40)
    parser.add_argument("--skip-profile", action="store_true", help="Reuse existing profiled CSV datasets.")
    parser.add_argument(
        "--sample-count",
        action="append",
        default=[],
        help="Override sample count, e.g. conv-bn-relu=350",
    )
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=None,
        help="Kernels to train (default: conv-bn-relu, fc, avgpool, maxpool).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Resume an existing run directory instead of creating a new timestamped folder.",
    )
    return parser.parse_args()


def parse_sample_overrides(items):
    overrides = {}
    for item in items:
        kernel, count = item.split("=")
        overrides[kernel.strip()] = int(count.strip())
    return overrides


def config_to_features(kernel_type: str, config: dict) -> list[float]:
    cfg = dict(config)
    if "COUT" not in cfg and kernel_type.startswith("dwconv"):
        cfg["COUT"] = cfg["CIN"]
    if kernel_type in {"avgpool", "maxpool"}:
        if "STRIDES" not in cfg and "POOL_STRIDES" in cfg:
            cfg["STRIDES"] = cfg["POOL_STRIDES"]
        if "COUT" not in cfg:
            cfg["COUT"] = cfg["CIN"]

    if kernel_type in {"conv-bn-relu", "dwconv-bn-relu", "fc"}:
        cols = FEATURE_COLUMNS[kernel_type][: len(FEATURE_COLUMNS[kernel_type]) - 2]
        feature = [cfg[c] for c in cols]
        flops, params = get_flops_params(kernel_type, cfg)
        feature.extend([flops / 2e6, params / 1e6])
        return feature

    cols = FEATURE_COLUMNS[kernel_type]
    return [cfg[c] for c in cols]


def sample_configs(kernel_type: str, sample_num: int):
    random.seed(10)
    np.random.seed(10)
    if kernel_type == "fc":
        sample_num = max(sample_num, 4)
    return get_sampler_for_kernel(kernel_type, sample_num, "prior")


def convert_keras_to_tflite(keras_path: Path, output_dir: Path) -> Path:
    import shutil
    import tensorflow as tf

    model_name = keras_path.name
    if keras_path.is_dir():
        converter = tf.lite.TFLiteConverter.from_saved_model(str(keras_path))
    else:
        model = tf.keras.models.load_model(str(keras_path))
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = output_dir / f"{model_name}.tflite"
    tflite_path.write_bytes(tflite_model)
    if keras_path.is_dir():
        shutil.rmtree(keras_path, ignore_errors=True)
    elif keras_path.is_file():
        keras_path.unlink(missing_ok=True)
    return tflite_path


def generate_models(kernel_type: str, configs: list[dict], kernel_dir: Path):
    kernel_dir.mkdir(parents=True, exist_ok=True)
    records = []
    errors = []

    for idx, config in enumerate(configs):
        random_id = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        keras_path = kernel_dir / f"{kernel_type}_prior_{random_id}"
        try:
            _, _, saved_config = generate_model_for_kernel(
                kernel_type,
                config,
                save_path=str(keras_path),
                implement="tensorflow",
                batch_size=1,
            )
            model_path = convert_keras_to_tflite(keras_path, kernel_dir)
            records.append(
                {
                    "id": random_id,
                    "model_path": str(model_path),
                    "config": saved_config,
                }
            )
        except Exception as exc:
            errors.append({"config": config, "error": str(exc)})

    return records, errors


def profile_models(records, adb, args):
    profiled = []
    failures = []
    for idx, record in enumerate(records, start=1):
        model_path = record["model_path"]
        try:
            std_ms, avg_ms, _ = run_on_android(
                model_path,
                adb,
                remote_dir=args.remote_model_dir,
                benchmark_path=args.benchmark_path,
                taskset=args.taskset,
                warm_ups=args.warm_ups,
                num_runs=args.num_runs,
            )
            if avg_ms <= 0:
                raise RuntimeError(f"Invalid latency avg={avg_ms}")
            profiled.append(
                {
                    **record,
                    "latency_ms": avg_ms,
                    "latency_std_ms": std_ms,
                }
            )
            print(f"[{idx}/{len(records)}] {Path(model_path).name}: {avg_ms:.4f} ms")
        except Exception as exc:
            failures.append({"model_path": model_path, "error": str(exc)})
            print(f"[{idx}/{len(records)}] FAILED {Path(model_path).name}: {exc}")
    return profiled, failures


def build_dataset(kernel_type: str, profiled_records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in profiled_records:
        features = config_to_features(kernel_type, record["config"])
        row = dict(zip(FEATURE_COLUMNS[kernel_type], features))
        row["LATENCY"] = record["latency_ms"]
        row["model_path"] = record["model_path"]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_predictor(model, dataloader):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            preds.extend(pred.cpu().numpy().tolist())
            targets.extend(y.cpu().numpy().tolist())
    rmse, rmspe, error, acc5, acc10, acc15 = latency_metrics(np.array(preds), np.array(targets))
    return {
        "rmse": float(rmse),
        "rmspe": float(rmspe),
        "relative_error": float(error),
        "acc5": float(acc5),
        "acc10": float(acc10),
        "acc15": float(acc15),
    }


def train_kernel_predictor(
    kernel_type: str,
    dataset_csv: Path,
    output_dir: Path,
    predictor_pool: Path,
    args,
):
    feature_cols = FEATURE_COLUMNS[kernel_type]
    df = pd.read_csv(dataset_csv)
    x_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df["LATENCY"].to_numpy(dtype=np.float32)

    x_train, x_eval, y_train, y_eval = train_test_split(
        x_all,
        y_all,
        test_size=args.test_size,
        random_state=args.seed,
    )

    train_dataset = Data.TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    eval_dataset = Data.TensorDataset(torch.tensor(x_eval), torch.tensor(y_eval))

    transfer_device = None
    transfer_weights = None
    pool_path = Path(predictor_pool)
    if pool_path.exists() and any(pool_path.iterdir()):
        detector_dataset = Data.TensorDataset(
            torch.tensor(x_all[: min(200, len(x_all))]),
            torch.tensor(y_all[: min(200, len(y_all))]),
        )
        detector = Detector(pool_path=str(pool_path), dataset=detector_dataset, kernel_type=kernel_type)
        similar_devices = detector.get_similar_device()
        if similar_devices:
            transfer_device = similar_devices[0]
            weight_path = pool_path / transfer_device / f"{kernel_type}.pth"
            if weight_path.exists():
                transfer_weights = torch.load(weight_path, map_location="cpu")

    predictor_dir = output_dir / "predictors"
    predictor_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        kernel_type=kernel_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        save_dir=str(predictor_dir),
        weights=transfer_weights,
    )
    trainer.train()
    trainer.save()

    train_metrics = evaluate_predictor(
        trainer.model,
        trainer.get_dataloader(train_dataset),
    )
    eval_metrics = evaluate_predictor(
        trainer.model,
        trainer.get_dataloader(eval_dataset),
    )

    return {
        "kernel_type": kernel_type,
        "num_samples_total": int(len(df)),
        "num_train": int(len(x_train)),
        "num_eval": int(len(x_eval)),
        "transfer_device": transfer_device,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "predictor_path": str(predictor_dir / f"{kernel_type}.pth"),
        "dataset_path": str(dataset_csv),
    }


def main():
    args = parse_args()
    kernels = args.kernels if args.kernels else KERNELS
    for kernel_type in kernels:
        if kernel_type not in FEATURE_COLUMNS:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
    sample_overrides = parse_sample_overrides(args.sample_count)
    sample_counts = {**DEFAULT_SAMPLE_COUNTS, **sample_overrides}

    if args.run_dir:
        output_dir = Path(args.run_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_root) / f"pixel9_tflite_cpu_{timestamp}"
    datasets_dir = output_dir / "datasets"
    kernels_dir = output_dir / "kernels"
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    adb = ADBConnect(serial=args.device_serial)
    device_info = adb.run_cmd("getprop ro.product.model").strip()
    android_release = adb.run_cmd("getprop ro.build.version.release").strip()

    report = {
        "platform": "Pixel 9 CPU",
        "device_serial": adb.serial,
        "device_model": device_info,
        "android_release": android_release,
        "framework": "TFLite (benchmark_model on CPU, XNNPACK delegate)",
        "benchmark_path": args.benchmark_path,
        "remote_model_dir": args.remote_model_dir,
        "taskset": args.taskset,
        "predictor_pool": str(Path(args.predictor_pool)),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "test_size": args.test_size,
            "seed": args.seed,
            "warm_ups": args.warm_ups,
            "num_runs": args.num_runs,
        },
        "sample_counts_requested": sample_counts,
        "kernels": {},
        "assumptions": [
            "LitePred MLP transfer-learning pipeline is used (Detector + Trainer).",
            "Kernel TFLite models are generated with nn-Meter TensorFlow kernel blocks because LitePred does not ship a standalone model generator.",
            "On-device profiling uses LitePred/profile_script with non-root adb shell (Pixel 9 has no su).",
            "CPU inference is pinned with taskset mask 60 and --num_threads=1.",
            "Latency label is benchmark_model avg inference time in milliseconds.",
            "Train/validation split uses sklearn train_test_split(test_size=0.2, random_state=10).",
        ],
        "started_at": timestamp,
    }

    t0 = time.time()
    for kernel_type in kernels:
        print(f"\n=== {kernel_type} ===")
        dataset_csv = datasets_dir / f"Data_{kernel_type}_litepred.csv"

        if args.skip_profile and dataset_csv.exists():
            print(f"Reusing existing dataset: {dataset_csv}")
            profile_stats = {"reused_existing_dataset": True}
        else:
            sample_num = sample_counts[kernel_type]
            configs = sample_configs(kernel_type, sample_num)
            records, gen_errors = generate_models(kernel_type, configs, kernels_dir / kernel_type)
            profiled, profile_errors = profile_models(records, adb, args)
            df = build_dataset(kernel_type, profiled)
            if df.empty:
                raise RuntimeError(
                    f"No successful profile results for {kernel_type}. "
                    f"See {output_dir / f'profile_errors_{kernel_type}.json'}"
                )
            df.to_csv(dataset_csv, index=False)
            profile_stats = {
                "configs_sampled": len(configs),
                "models_generated": len(records),
                "models_profiled": len(profiled),
                "generation_failures": len(gen_errors),
                "profiling_failures": len(profile_errors),
            }
            (output_dir / f"profile_errors_{kernel_type}.json").write_text(
                json.dumps({"generation": gen_errors, "profiling": profile_errors}, indent=2)
            )

        kernel_report = train_kernel_predictor(
            kernel_type,
            dataset_csv,
            output_dir,
            args.predictor_pool,
            args,
        )
        kernel_report["profiling"] = profile_stats
        report["kernels"][kernel_type] = kernel_report
        print(json.dumps(kernel_report["eval_metrics"], indent=2))

    report["elapsed_seconds"] = round(time.time() - t0, 2)
    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved training report to {report_path}")
    print(f"Predictors saved under {output_dir / 'predictors'}")


if __name__ == "__main__":
    main()
