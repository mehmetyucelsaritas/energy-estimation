from nn_meter.builder import builder_config
from nn_meter.builder.nn_meter_builder import build_latency_predictor

builder_config.init("./nn_meter_onnx_workspace")
build_latency_predictor(backend="onnx_mac_m4")