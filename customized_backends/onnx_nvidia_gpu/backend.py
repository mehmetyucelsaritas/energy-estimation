# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Local ONNX Runtime backend for nn-Meter targeting NVIDIA GPU (CUDA EP)."""

import logging
import os
import site
import sys

from nn_meter.builder.backends import BaseBackend

_pkg = os.path.dirname(os.path.abspath(__file__))
_customized = os.path.dirname(_pkg)
if _customized not in sys.path:
    sys.path.insert(0, _customized)

from onnx_mac_m4.backend import ONNXMacBackend

logging = logging.getLogger("nn-Meter")


def _ensure_nvidia_pip_cuda_libs_on_ld_path():
    """Let ONNX Runtime's CUDA EP find libcublasLt / cuDNN / curand from pip `nvidia-*-cu12` wheels.

    onnxruntime-gpu 1.19+ expects CUDA 12.x and cuDNN 9.x at load time. A matching PyTorch wheel
    can satisfy that, but older stacks (e.g. torch+cu102) do not; install
    ``docs/requirements/onnxruntime_gpu_cuda12_linux.txt`` and this prepends those ``lib`` dirs
    before the first ``import onnxruntime``.
    """
    bases = []
    if hasattr(site, "getsitepackages"):
        bases.extend(site.getsitepackages())
    user_sp = getattr(site, "getusersitepackages", lambda: None)()
    if user_sp:
        bases.append(user_sp)
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    bases.append(os.path.join(sys.prefix, "lib", f"python{ver}", "site-packages"))

    lib_dirs = []
    seen = set()
    for base in bases:
        if not base or not os.path.isdir(base):
            continue
        nvidia_root = os.path.join(base, "nvidia")
        if not os.path.isdir(nvidia_root):
            continue
        for name in sorted(os.listdir(nvidia_root)):
            lib = os.path.join(nvidia_root, name, "lib")
            if os.path.isdir(lib) and lib not in seen:
                seen.add(lib)
                lib_dirs.append(lib)
    if not lib_dirs:
        return
    prefix = ":".join(lib_dirs)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = prefix + (":" + cur if cur else "")


_ensure_nvidia_pip_cuda_libs_on_ld_path()


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _resolve_nvidia_providers(c):
    import onnxruntime as ort

    explicit = c.get("EXECUTION_PROVIDERS")
    if explicit is not None:
        if isinstance(explicit, str):
            return [p.strip() for p in explicit.split(",") if p.strip()]
        if isinstance(explicit, list):
            return [str(p).strip() for p in explicit if str(p).strip()]
    available = set(ort.get_available_providers())
    chain = []
    if _truthy(c.get("USE_TENSORRT_EP")) and "TensorrtExecutionProvider" in available:
        chain.append("TensorrtExecutionProvider")
    if _truthy(c.get("USE_CUDA_EP", True)) and "CUDAExecutionProvider" in available:
        chain.append("CUDAExecutionProvider")
    chain.append("CPUExecutionProvider")
    seen = set()
    out = []
    for p in chain:
        if p in available and p not in seen:
            out.append(p)
            seen.add(p)
    return out or ["CPUExecutionProvider"]


class OnnxNvidiaGpuBackend(ONNXMacBackend):
    """Host-side profiling via ONNX Runtime with CUDA (and optional TensorRT) EP."""

    def update_configs(self):
        BaseBackend.update_configs(self)
        c = self.configs or {}
        providers = _resolve_nvidia_providers(c)
        batch = c.get("BATCH_SIZE", c.get("DYNAMIC_BATCH_DIM", 1))
        self.profiler_kwargs.update(
            {
                "warmup_runs": c.get("WARMUP_RUNS", 10),
                "num_runs": c.get("NUM_RUNS", 50),
                "providers": providers,
                "dynamic_batch_dim": batch,
                "intra_op_num_threads": c.get("INTRA_OP_NUM_THREADS", 0),
                "inter_op_num_threads": c.get("INTER_OP_NUM_THREADS", 0),
                "verbose": c.get("VERBOSE", False),
                "power_fast_legacy_delta": _truthy(c.get("POWER_FAST_LEGACY_DELTA", False)),
            }
        )

    def test_connection(self):
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            logging.warning(
                "CUDAExecutionProvider not in onnxruntime.get_available_providers(); "
                f"got {providers}. Install a GPU build of onnxruntime (e.g. onnxruntime-gpu) for NVIDIA profiling."
            )
        logging.keyinfo("hello backend !")
