#!/usr/bin/env bash
set -euo pipefail

cd /workspace/energy-estimation

backend_meta_file="/tmp/onnx_nvidia_gpu_backend_meta.yaml"
cat > "${backend_meta_file}" <<'EOF'
builtin_name: onnx_nvidia_gpu
package_location: /workspace/energy-estimation/customized_backends/onnx_nvidia_gpu
class_module: backend
class_name: OnnxNvidiaGpuBackend
defaultConfigFile: /workspace/energy-estimation/customized_backends/onnx_nvidia_gpu/default_config.yaml
EOF

python -m nn_meter.utils.nn_meter_cli.interface register --backend "${backend_meta_file}"
python build_nvidia_latency_predictor.py
