"""Build power predictors using onnx_mac_m4 batch profiling (one CodeCarbon session per chunk).

Per-model power uses CodeCarbon checkpoint segments when available. For that API, use a
recent CodeCarbon (e.g. ``PYTHONPATH=/path/to/codecarbon-master`` or ``pip install -e``).
"""
from nn_meter.builder import builder_config
from nn_meter.builder.nn_meter_builder import build_power_predictor

builder_config.init("./nn_meter_onnx_workspace")
build_power_predictor(backend="onnx_mac_m4")
