#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import csv
import time
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchvision.models as models
from nn_meter import load_latency_predictor


TORCHVISION_MODEL_ZOO = {
    "resnet18": "models.resnet18()",
    "alexnet": "models.alexnet()",
    "vgg16": "models.vgg16()",
    "squeezenet": "models.squeezenet1_0()",
    "densenet161": "models.densenet161()",
    "inception_v3": "models.inception_v3()",
    "googlenet": "models.googlenet()",
    "shufflenet_v2": "models.shufflenet_v2_x1_0()",
    "mobilenet_v2": "models.mobilenet_v2()",
    "resnext50_32x4d": "models.resnext50_32x4d()",
    "wide_resnet50_2": "models.wide_resnet50_2()",
    "mnasnet": "models.mnasnet1_0()",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict full-network latency for torchvision models using a custom nn-Meter predictor."
    )
    parser.add_argument(
        "--predictor",
        default="onnx_workspace_latency",
        help="Registered predictor name (default: onnx_workspace_latency).",
    )
    parser.add_argument(
        "--predictor-version",
        type=float,
        default=1.0,
        help="Registered predictor version (default: 1.0).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(TORCHVISION_MODEL_ZOO.keys()),
        help='Model names to run, or "all".',
    )
    parser.add_argument(
        "--output-csv",
        default="tests/integration_test/custom_predictor_torch_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--apply-nni",
        action="store_true",
        default=False,
        help="Use NNI-based torch converter.",
    )
    parser.add_argument(
        "--no-apply-nni",
        dest="apply_nni",
        action="store_false",
        help="Use ONNX-based torch converter (default).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue predicting remaining models even if one model fails.",
    )
    parser.add_argument(
        "--fail-fast",
        dest="continue_on_error",
        action="store_false",
        help="Stop immediately when a model prediction fails.",
    )
    parser.add_argument(
        "--runtime-warmup",
        type=int,
        default=10,
        help="Warmup iterations for measured runtime latency.",
    )
    parser.add_argument(
        "--runtime-runs",
        type=int,
        default=30,
        help="Timed iterations for measured runtime latency.",
    )
    parser.add_argument(
        "--runtime-device",
        choices=["cpu", "mps"],
        default="cpu",
        help="Device for runtime latency measurement (default: cpu).",
    )
    return parser.parse_args()


def get_model_names(requested):
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(TORCHVISION_MODEL_ZOO.keys())
    unknown = [m for m in requested if m not in TORCHVISION_MODEL_ZOO]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Supported: {list(TORCHVISION_MODEL_ZOO.keys())}")
    return requested


def measure_runtime_latency_ms(model, model_name, device, warmup_runs, runs):
    model.eval()
    input_tensor = torch.randn(1, 3, 224, 224)
    if model_name == "inception_v3":
        input_tensor = torch.randn(1, 3, 299, 299)

    # Align runtime measurement backend with predictor-building backend family:
    # export torch model to ONNX and benchmark with ONNX Runtime providers.
    available = ort.get_available_providers()
    if device == "mps":
        requested_providers = (
            ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            if "CoreMLExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
    else:
        requested_providers = ["CPUExecutionProvider"]

    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        torch.onnx.export(
            model,
            input_tensor,
            f.name,
            export_params=True,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            opset_version=12,
        )
        sess = ort.InferenceSession(f.name, providers=requested_providers)
        ort_input = {sess.get_inputs()[0].name: input_tensor.numpy().astype(np.float32)}

        for _ in range(warmup_runs):
            _ = sess.run(None, ort_input)

        start = time.perf_counter()
        for _ in range(runs):
            _ = sess.run(None, ort_input)
        elapsed_s = time.perf_counter() - start

    used_providers = sess.get_providers()
    runtime_provider_used = used_providers[0] if used_providers else "UNKNOWN"
    return {
        "runtime_latency_ms": (elapsed_s / runs) * 1000.0,
        "runtime_provider_used": runtime_provider_used,
        "runtime_providers_requested": ",".join(requested_providers),
        "runtime_providers_available": ",".join(available),
    }


def main():
    args = parse_args()
    model_names = get_model_names(args.models)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    predictor = load_latency_predictor(args.predictor, args.predictor_version)
    print(f"ONNX Runtime available providers: {ort.get_available_providers()}")
    rows = []

    for model_name in model_names:
        model = eval(TORCHVISION_MODEL_ZOO[model_name])
        try:
            # This predicts latency for the entire network graph after conversion/splitting.
            pred_latency_ms = predictor.predict(model, "torch", apply_nni=args.apply_nni)
            runtime_result = measure_runtime_latency_ms(
                model,
                model_name,
                args.runtime_device,
                args.runtime_warmup,
                args.runtime_runs,
            )
            runtime_latency_ms = runtime_result["runtime_latency_ms"]
            rows.append(
                (
                    model_name,
                    "torch",
                    args.predictor,
                    args.predictor_version,
                    args.runtime_device,
                    runtime_result["runtime_providers_requested"],
                    runtime_result["runtime_provider_used"],
                    runtime_result["runtime_providers_available"],
                    float(pred_latency_ms),
                    float(runtime_latency_ms),
                    float(pred_latency_ms) - float(runtime_latency_ms),
                )
            )
            print(
                f"{model_name}: predicted={float(pred_latency_ms):.4f} ms, "
                f"runtime={float(runtime_latency_ms):.4f} ms, "
                f"diff={float(pred_latency_ms) - float(runtime_latency_ms):.4f} ms, "
                f"provider_used={runtime_result['runtime_provider_used']}"
            )
        except Exception as exc:
            print(f"{model_name}: ERROR ({exc})")
            if not args.continue_on_error:
                raise

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_name",
                "model_type",
                "predictor",
                "predictor_version",
                "runtime_device",
                "runtime_providers_requested",
                "runtime_provider_used",
                "runtime_providers_available",
                "predicted_latency_ms",
                "runtime_latency_ms",
                "diff_ms_pred_minus_runtime",
            ]
        )
        writer.writerows(rows)

    print(f"Saved results to: {output_csv}")


if __name__ == "__main__":
    main()
