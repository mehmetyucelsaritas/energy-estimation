# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Local ONNX Runtime backend for nn-Meter targeting NVIDIA GPU (CUDA EP)."""

import logging
import os
import site
import sys
import glob
import ctypes

from nn_meter.builder.backends import BaseBackend

_pkg = os.path.dirname(os.path.abspath(__file__))
_customized = os.path.dirname(_pkg)
if _customized not in sys.path:
    sys.path.insert(0, _customized)

from onnx_mac_m4.backend import ONNXMacBackend

logging = logging.getLogger("nn-Meter")


def _collect_nvidia_pip_cuda_lib_dirs():
    """Collect nvidia/*/lib directories from active Python site-packages."""
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
    return lib_dirs


def _prepend_on_linux_ld_library_path(lib_dirs):
    if not lib_dirs or sys.platform != "linux":
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    cur_entries = [p for p in cur.split(":") if p]
    merged = []
    seen = set()
    for p in lib_dirs + cur_entries:
        if p not in seen:
            seen.add(p)
            merged.append(p)
    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


def _preload_nvidia_cuda_libs(lib_dirs):
    """Preload CUDA libs so ORT CUDA EP can resolve dependencies in-process.

    Updating LD_LIBRARY_PATH after process start can be too late for dynamic linking on Linux.
    Loading likely dependencies via absolute paths avoids repeated CUDAExecutionProvider failures.
    """
    if not lib_dirs or sys.platform != "linux":
        return
    patterns = [
        "libcublasLt.so*",
        "libcublas.so*",
        "libcudnn.so*",
        "libcurand.so*",
        "libcufft.so*",
        "libcudart.so*",
        "libnvrtc.so*",
    ]
    candidates = []
    seen = set()
    for pattern in patterns:
        for d in lib_dirs:
            for path in sorted(glob.glob(os.path.join(d, pattern)), reverse=True):
                if path not in seen:
                    seen.add(path)
                    candidates.append(path)
    if not candidates:
        return
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for path in candidates:
        try:
            ctypes.CDLL(path, mode=mode)
        except OSError:
            # Best effort: some entries are symlinks or version variants not present in this stack.
            continue


def _ensure_nvidia_pip_cuda_libs_on_ld_path():
    """Expose pip CUDA libs (cu11/cu12) for ONNX Runtime CUDA/TensorRT providers."""
    lib_dirs = _collect_nvidia_pip_cuda_lib_dirs()
    if not lib_dirs:
        return
    _prepend_on_linux_ld_library_path(lib_dirs)
    _preload_nvidia_cuda_libs(lib_dirs)


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
