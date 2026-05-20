#!/usr/bin/env python3
"""Run Pixel 9 CPU latency/energy pipeline with single-pass profiling.

Stages:
1) (optional) fusion-rule testcase generation + profiling + rule detection
2) (optional) latency+energy predictor build with shared profiling runs
"""

import argparse
import os
import sys
from pathlib import Path

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BACKEND = os.environ.get("NN_METER_BACKEND", "pixel9_cpu_idp")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _init_builder():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from nn_meter.builder import builder_config

    os.chdir(WORKSPACE)
    builder_config.init(WORKSPACE)


def _run_fusion_rule_stage(backend_name: str):
    from nn_meter.builder import profile_models, builder_config
    from nn_meter.builder.backend_meta.fusion_rule_tester import (
        detect_fusion_rule,
        generate_testcases,
    )
    from nn_meter.builder.backends import connect_backend

    print("(1/2) Generate fusion-rule testcases...")
    generate_testcases()

    ruletest_dir = builder_config.get("WORKSPACE", "ruletest")
    origin = os.path.join(ruletest_dir, "results", "origin_testcases.json")

    print(f"(1/2) Profile fusion testcases on backend '{backend_name}'...")
    backend = connect_backend(backend_name=backend_name)
    profile_models(backend, origin, mode="ruletest")

    profiled = os.path.join(ruletest_dir, "results", "profiled_results.json")
    print("(1/2) Detect fusion rules...")
    detect_fusion_rule(profiled)

    rules = os.path.join(ruletest_dir, "results", "detected_fusion_rule.json")
    print(f"Fusion-rule stage complete. Output: {rules}")


def _run_predictor_stage(backend_name: str):
    from nn_meter.builder import build_latency_energy_predictor_single_pass

    print(f"(2/2) Build latency+energy predictors (single pass) on backend '{backend_name}'...")
    build_latency_energy_predictor_single_pass(backend=backend_name)
    print("(2/2) Predictor stage complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Pixel 9 CPU latency/energy pipeline with single-pass profiling."
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"Backend name (default: {DEFAULT_BACKEND})",
    )
    parser.add_argument(
        "--skip-fusion",
        action="store_true",
        help="Skip fusion-rule stage.",
    )
    parser.add_argument(
        "--skip-predictor",
        action="store_true",
        help="Skip predictor build stage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.skip_fusion and args.skip_predictor:
        raise ValueError("Both stages are skipped. Remove at least one skip flag.")

    _init_builder()

    if not args.skip_fusion:
        _run_fusion_rule_stage(args.backend)
    else:
        print("(1/2) Fusion-rule stage skipped.")

    if not args.skip_predictor:
        _run_predictor_stage(args.backend)
    else:
        print("(2/2) Predictor stage skipped.")

    print("Pipeline finished.")


if __name__ == "__main__":
    main()
