#!/usr/bin/env python3
"""Resume fusion-rule profiling for missing latency entries only."""

import json
import os
import sys
from pathlib import Path

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.environ.get("NN_METER_BACKEND", "pixel9_cpu_idp")
REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _missing_entries(profiled):
    missing = []
    for module, entries in profiled.items():
        for entry_id, entry in entries.items():
            if not entry or "latency" not in entry:
                missing.append((module, entry_id))
    return missing


def _build_missing_models(origin, profiled):
    missing_models = {}
    for module, entry_id in _missing_entries(profiled):
        model_info = origin[module][entry_id]
        missing_models.setdefault(module, {})[entry_id] = {
            "model": model_info["model"],
            "shapes": model_info["shapes"],
        }
    return missing_models


def _check_device_ready():
    import subprocess

    output = subprocess.check_output(["adb", "devices"], text=True, encoding="utf-8", errors="replace")
    lines = [line.strip() for line in output.splitlines() if line.strip() and not line.startswith("List of devices")]
    if not lines:
        raise RuntimeError("No ADB device connected. Connect Pixel 9 and rerun.")
    if any("\tunauthorized" in line for line in lines):
        raise RuntimeError(
            "ADB device is unauthorized. Accept the USB debugging prompt on Pixel 9, then rerun."
        )
    if all("\toffline" in line for line in lines):
        raise RuntimeError("ADB device is offline. Reconnect Pixel 9 and rerun.")


def main():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from nn_meter.builder import builder_config, profile_models
    from nn_meter.builder.backend_meta.fusion_rule_tester import detect_fusion_rule
    from nn_meter.builder.backends import connect_backend

    os.chdir(WORKSPACE)
    builder_config.init(WORKSPACE)

    ruletest_dir = builder_config.get("WORKSPACE", "ruletest")
    origin_path = os.path.join(ruletest_dir, "results", "origin_testcases.json")
    profiled_path = os.path.join(ruletest_dir, "results", "profiled_results.json")

    origin = _load_json(origin_path)
    profiled = _load_json(profiled_path)
    missing_before = _missing_entries(profiled)
    print(f"Missing latency entries before resume: {len(missing_before)}")
    for module, entry_id in missing_before:
        print(f"  - {module}/{entry_id}")

    if not missing_before:
        print("Nothing to resume.")
    else:
        _check_device_ready()
        missing_models = _build_missing_models(origin, profiled)
        print(f"Profiling {len(missing_before)} missing testcases on backend '{BACKEND}'...")
        backend = connect_backend(backend_name=BACKEND)
        profile_models(
            backend,
            missing_models,
            mode="ruletest",
            save_name="profiled_results.json",
        )

    profiled = _load_json(profiled_path)
    missing_after = _missing_entries(profiled)
    print(f"Missing latency entries after resume: {len(missing_after)}")
    for module, entry_id in missing_after:
        print(f"  - {module}/{entry_id}")

    if missing_after:
        print("Some entries are still missing. Check fusion_rule_test/results/profile_error.log")
        return

    print("Detecting fusion rules from updated profiled results...")
    detect_fusion_rule(profiled_path)

    rules = os.path.join(ruletest_dir, "results", "detected_fusion_rule.json")
    print(f"Done. Fusion rules saved at: {rules}")


if __name__ == "__main__":
    main()
