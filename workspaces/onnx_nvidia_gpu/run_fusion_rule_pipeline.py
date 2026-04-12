#!/usr/bin/env python3
"""Fusion rule detection for docs/builder/test_fusion_rules.md (end-to-end demo).

Uses this workspace and the registered onnx_nvidia_gpu backend by default.

  python run_fusion_rule_pipeline.py

Override backend:
  NN_METER_BACKEND=onnx_mac_m4 python run_fusion_rule_pipeline.py

Requires torch (IMPLEMENT: torch), onnxruntime, and a registered backend. Use the project conda
env and install this repo in editable mode so torch FC/reshape test cases match current code:

  conda activate energy-estimation
  pip install -e /path/to/energy-estimation
  python run_fusion_rule_pipeline.py

For GPU: pip install -r ../../docs/requirements/onnxruntime_gpu_cuda12_linux.txt
"""
import os
import site
import sys

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("NN_METER_BACKEND", "onnx_nvidia_gpu")


def _ensure_nvidia_pip_cuda_libs_on_ld_path():
    """Must run before any ``import onnxruntime`` so CUDA EP finds libcublasLt.so.12 (pip nvidia-*-cu12)."""
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
        root = os.path.join(base, "nvidia")
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            lib = os.path.join(root, name, "lib")
            if os.path.isdir(lib) and lib not in seen:
                seen.add(lib)
                lib_dirs.append(lib)
    if not lib_dirs:
        return False
    prefix = ":".join(lib_dirs)
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = prefix + (":" + cur if cur else "")
    return True


def main():
    if BACKEND == "onnx_nvidia_gpu" and not _ensure_nvidia_pip_cuda_libs_on_ld_path():
        print(
            "Note: no pip NVIDIA CUDA 12 libs found under site-packages/nvidia/*/lib; "
            "CUDA EP will likely fail. Install:\n"
            "  pip install -r ../../docs/requirements/onnxruntime_gpu_cuda12_linux.txt\n"
        )
    from nn_meter.builder import builder_config, profile_models
    from nn_meter.builder.backends import connect_backend
    from nn_meter.builder.backend_meta.fusion_rule_tester import (
        generate_testcases,
        detect_fusion_rule,
    )

    os.chdir(WORKSPACE)
    builder_config.init(WORKSPACE)

    print("(1) Generate test cases → fusion_rule_test/models/, results/origin_testcases.json")
    generate_testcases()

    fusion_dir = builder_config.get("WORKSPACE", "ruletest")
    origin = os.path.join(fusion_dir, "results", "origin_testcases.json")

    print(f"(2) Profile models on backend {BACKEND!r} → results/profiled_results.json")
    backend = connect_backend(backend_name=BACKEND)
    profile_models(backend, origin, mode="ruletest")

    profiled = os.path.join(fusion_dir, "results", "profiled_results.json")
    print("(3) Detect fusion rules → results/detected_fusion_rule.json")
    detect_fusion_rule(profiled)

    rules_path = os.path.join(fusion_dir, "results", "detected_fusion_rule.json")
    print("Done.")
    print(f"    Detected rules: {rules_path}")


if __name__ == "__main__":
    main()
