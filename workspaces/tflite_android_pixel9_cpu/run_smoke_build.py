#!/usr/bin/env python3
"""Smoke-test build for maxpool latency+energy predictors (single profiling pass) on Pixel 9 CPU."""

import os
import sys
from pathlib import Path

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("NN_METER_BACKEND", "pixel9_cpu_idp")
REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from nn_meter.builder import build_latency_energy_predictor_single_pass, builder_config

    os.chdir(WORKSPACE)
    builder_config.init(WORKSPACE)

    print(f"Building latency+energy predictors with single profiling pass on backend '{BACKEND}'...")
    build_latency_energy_predictor_single_pass(backend=BACKEND)

    print("Smoke build complete.")


if __name__ == "__main__":
    main()
