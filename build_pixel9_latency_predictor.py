from pathlib import Path

from nn_meter.builder import builder_config
from nn_meter.builder.nn_meter_builder import build_latency_predictor


workspace_dir = Path(__file__).resolve().parent / "workspaces" / "tflite_android"
builder_config.init(str(workspace_dir))
build_latency_predictor(backend="tflite_cpu")
