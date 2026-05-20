#!/usr/bin/env python3
"""Fusion-rule testing pipeline for Pixel 9 CPU IDP backend."""

import os
import sys
from pathlib import Path

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("NN_METER_BACKEND", "pixel9_cpu_idp")
REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from nn_meter.builder import builder_config, profile_models
    from nn_meter.builder.backend_meta.fusion_rule_tester import (
        detect_fusion_rule,
        generate_testcases,
    )
    from nn_meter.builder.backends import connect_backend

    os.chdir(WORKSPACE)
    builder_config.init(WORKSPACE)

    print("(1) Generate fusion-rule testcases...")
    generate_testcases()

    ruletest_dir = builder_config.get("WORKSPACE", "ruletest")
    origin = os.path.join(ruletest_dir, "results", "origin_testcases.json")

    print(f"(2) Profile testcases on backend '{BACKEND}'...")
    backend = connect_backend(backend_name=BACKEND)
    profile_models(backend, origin, mode="ruletest")

    profiled = os.path.join(ruletest_dir, "results", "profiled_results.json")
    print("(3) Detect fusion rules...")
    detect_fusion_rule(profiled)

    rules = os.path.join(ruletest_dir, "results", "detected_fusion_rule.json")
    print(f"Done. Fusion rules saved at: {rules}")


if __name__ == "__main__":
    main()
