# Pixel 9 CPU Workspace (IDP-integrated)

This workspace follows the nn-Meter builder workflow, but uses the custom backend
`pixel9_cpu_idp` so model profiling on device reports:

- `latency` (ms per inference)
- `energy` (mJ per inference from Perfetto rails)

The backend runs ONNX models with `IDP/device_runner/run_onnx_arm64` on Pixel 9 and
uses **Perfetto only** for energy extraction.

## 1) Backend preparation

1. Connect Pixel 9 with USB debugging enabled and authorized:
   - `adb devices`
2. Ensure IDP runner files exist locally:
   - `IDP/device_runner/run_onnx_arm64`
   - `IDP/device_runner/ort_android/jni/arm64-v8a/libonnxruntime.so`
   - `IDP/device_runner/ort_android/jni/arm64-v8a/libc++_shared.so`
3. Verify backend config:
   - `configs/backend_config.yaml`

## 2) Fusion-rule testing

Run the documented flow from `docs/builder/test_fusion_rules.md`:

```bash
cd workspaces/tflite_android_pixel9_cpu
python run_fusion_rule_pipeline.py
```

Outputs:

- `fusion_rule_test/results/origin_testcases.json`
- `fusion_rule_test/results/profiled_results.json`
- `fusion_rule_test/results/detected_fusion_rule.json`

## 3) Predictor smoke build (maxpool only)

This workspace is preconfigured for the requested smoke settings:

```yaml
maxpool:
  INIT_SAMPLE_NUM: 100
  FINEGRAINED_SAMPLE_NUM: 20
  ITERATION: 1
  ERROR_THRESHOLD: 0.1
```

Run:

```bash
cd workspaces/tflite_android_pixel9_cpu
python run_smoke_build.py
```

Outputs:

- Latency predictors in `predictor_build/results/latency/predictors/`
- Energy predictors in `predictor_build/results/energy/predictors/`

## 4) Compatibility with customize predictor workflow

Use generated artifacts with `docs/builder/customize_predictor.md`:

- Fusion rules:
  - `fusion_rule_test/results/detected_fusion_rule.json`
- Kernel predictors:
  - `predictor_build/results/latency/predictors/*.pkl`
  - `predictor_build/results/energy/predictors/*.pkl`
