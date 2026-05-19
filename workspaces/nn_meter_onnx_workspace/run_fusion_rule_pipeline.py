#!/usr/bin/env python3
"""End-to-end fusion rule tester for nn_meter_onnx_workspace (see docs/builder/test_fusion_rules.md).

Run with your Anaconda env, e.g.:
  conda activate energy-estimation
  pip install -e /path/to/nn-Meter-main   # once
  python run_fusion_rule_pipeline.py
"""
import os

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("NN_METER_BACKEND", "onnx_mac_m4")


def main():
    from nn_meter.builder import builder_config, profile_models
    from nn_meter.builder.backends import connect_backend
    from nn_meter.builder.backend_meta.fusion_rule_tester import (
        generate_testcases,
        detect_fusion_rule,
    )
    from nn_meter.kernel_detector import KernelDetector

    os.chdir(WORKSPACE)
    builder_config.init(WORKSPACE)

    print("(1) Create test cases…")
    generate_testcases()

    fusion_dir = builder_config.get("WORKSPACE", "ruletest")
    origin = os.path.join(fusion_dir, "results", "origin_testcases.json")

    print(f"(2) Profile on backend {BACKEND!r}…")
    backend = connect_backend(backend_name=BACKEND)
    profile_models(backend, origin, mode="ruletest")

    profiled = os.path.join(fusion_dir, "results", "profiled_results.json")
    print("(3) Detect fusion rules…")
    detect_fusion_rule(profiled)

    rules_path = os.path.join(fusion_dir, "results", "detected_fusion_rule.json")
    print("(4) Load rules for kernel detection (KernelDetector)…")
    kd = KernelDetector(rules_path)
    print(
        f"    Fusible operator pairs from obey=True BF_* rules: {len(kd.reader.fusible)}; "
        f"sample: {kd.reader.fusible[:8]}"
    )
    print("Done.")
    print(f"    Rules JSON for predictors / kernel splitting: {rules_path}")


if __name__ == "__main__":
    main()
