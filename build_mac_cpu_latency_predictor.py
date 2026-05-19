from nn_meter.builder import builder_config
from nn_meter.builder.nn_meter_builder import build_latency_predictor


if __name__ == "__main__":
    builder_config.init("./nn_meter_onnx_cpu_workspace")
    build_latency_predictor(backend="onnx_mac_m4")
