#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import argparse
import csv
import tempfile
from collections import Counter
from pathlib import Path

import torch
import torchvision.models as models
from nn_meter import load_latency_predictor
from nn_meter.ir_converter import model_file_to_graph
from nn_meter.predictor.prediction.predict_by_kernel import merge_conv_kernels
from nn_meter.predictor.prediction.utils import get_kernel_name


TORCHVISION_MODEL_ZOO = {
    "resnet18": models.resnet18,
    "alexnet": models.alexnet,
    "vgg16": models.vgg16,
    "squeezenet": models.squeezenet1_0,
    "densenet161": models.densenet161,
    "inception_v3": models.inception_v3,
    "googlenet": models.googlenet,
    "shufflenet_v2": models.shufflenet_v2_x1_0,
    "mobilenet_v2": models.mobilenet_v2,
    "resnext50_32x4d": models.resnext50_32x4d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "mnasnet": models.mnasnet1_0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze kernels used per torchvision model for a given nn-Meter predictor."
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
        default=["all"],
        help='Model names to run, or "all".',
    )
    parser.add_argument(
        "--apply-nni",
        action="store_true",
        default=False,
        help="Reserved flag for compatibility. Ignored in this script.",
    )
    parser.add_argument(
        "--output-model-csv",
        default="tests/integration_test/model_kernel_map.csv",
        help="Output CSV path for model->kernel mapping.",
    )
    parser.add_argument(
        "--output-priority-csv",
        default="tests/integration_test/kernel_priority.csv",
        help="Output CSV path for global kernel priority.",
    )
    return parser.parse_args()


def get_model_names(requested):
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(TORCHVISION_MODEL_ZOO.keys())
    unknown = [m for m in requested if m not in TORCHVISION_MODEL_ZOO]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Supported: {list(TORCHVISION_MODEL_ZOO.keys())}")
    return requested


def main():
    args = parse_args()
    model_names = get_model_names(args.models)

    output_model_csv = Path(args.output_model_csv)
    output_priority_csv = Path(args.output_priority_csv)
    output_model_csv.parent.mkdir(parents=True, exist_ok=True)
    output_priority_csv.parent.mkdir(parents=True, exist_ok=True)

    predictor = load_latency_predictor(args.predictor, args.predictor_version)
    trained_kernel_names = set(predictor.kernel_predictors.keys())

    model_kernel_rows = []
    global_kernel_count = Counter()
    missing_kernel_count = Counter()

    for model_name in model_names:
        model = TORCHVISION_MODEL_ZOO[model_name]().eval()
        input_shape = (1, 3, 299, 299) if model_name == "inception_v3" else (1, 3, 224, 224)

        input_tensor = torch.randn(*input_shape)
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
            graph = model_file_to_graph(f.name, "onnx")
        predictor.kd.load_graph(graph)
        kernel_units = predictor.kd.get_kernels()

        model_counts = Counter()
        model_missing = Counter()

        # Mirror prediction-time op normalization.
        for item in kernel_units:
            op = item["op"]
            normalized_op = merge_conv_kernels(op)
            kernel_name = get_kernel_name(normalized_op)
            model_counts[kernel_name] += 1
            global_kernel_count[kernel_name] += 1
            if kernel_name not in trained_kernel_names:
                model_missing[kernel_name] += 1
                missing_kernel_count[kernel_name] += 1

        print(f"{model_name}: {sum(model_counts.values())} kernels, {sum(model_missing.values())} missing-predictor kernels")

        for kernel_name, count in sorted(model_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            model_kernel_rows.append(
                {
                    "model_name": model_name,
                    "kernel_name": kernel_name,
                    "kernel_count": count,
                    "is_missing_predictor": int(kernel_name not in trained_kernel_names),
                    "missing_count_in_model": model_missing.get(kernel_name, 0),
                }
            )

    with output_model_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "kernel_name",
                "kernel_count",
                "is_missing_predictor",
                "missing_count_in_model",
            ],
        )
        writer.writeheader()
        writer.writerows(model_kernel_rows)

    with output_priority_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kernel_name", "global_count", "missing_count"])
        for kernel_name, cnt in global_kernel_count.most_common():
            writer.writerow([kernel_name, cnt, missing_kernel_count.get(kernel_name, 0)])

    print(f"Saved model map to: {output_model_csv}")
    print(f"Saved priority table to: {output_priority_csv}")


if __name__ == "__main__":
    main()
