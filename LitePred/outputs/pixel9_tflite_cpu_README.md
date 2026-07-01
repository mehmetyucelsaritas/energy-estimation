# Pixel 9 CPU — LitePred Latency Predictor Training

End-to-end LitePred training for **Conv-BN-ReLU**, **FC**, **AvgPool**, and **MaxPool** on a connected **Pixel 9 (tokay) CPU** device using TFLite CPU inference.

## Run summary

| Item | Value |
|------|-------|
| Output directory | `LitePred/outputs/pixel9_tflite_cpu_20260608_194312/` |
| Device | Pixel 9 (`56040DLAQ004TS`), Android 16 |
| Total elapsed time | ~67 minutes (4026 s) |
| Environment | `energy-estimation` conda env (Python 3.8, PyTorch 2.4.1, TF 2.13.0) |

## How to reproduce

From repository root:

```bash
conda activate energy-estimation

# Download LitePred transfer-learning pool (one-time)
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
import os
repo = "fcq/pred_lite"
devices = [
    "pixel5tf27cpu_to_xiaomi12tf27cpu",
    "xiaomi11tf27cpu_to_pixel5tf27cpu",
    "xiaomi12tf21cpu_to_pixel6tf21cpu",
]
kernels = ["conv-bn-relu", "fc", "avgpool", "maxpool"]
root = "LitePred/predictors/pool"
for d in devices:
    for k in kernels:
        hf_hub_download(repo_id=repo, filename=f"{d}/{k}.pth", local_dir=root)
PY

python LitePred/scripts/train_pixel9_litepred.py \
  --device-serial 56040DLAQ004TS \
  --output-root LitePred/outputs
```

## Training configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr = 0.001 |
| Scheduler | CosineAnnealingLR (T_max = epochs) |
| Epochs | 350 |
| Batch size | 32 |
| Loss | MAPE (`torchmetrics.MeanAbsolutePercentageError`) |
| Train/val split | 80/20 (`random_state=10`) |
| Transfer learning | LitePred `Detector` + pretrained pool from [fcq/pred_lite](https://huggingface.co/fcq/pred_lite) |

### On-device profiling settings

| Parameter | Value |
|-----------|-------|
| Benchmark binary | `/data/local/tmp/benchmark_model` |
| Remote model path | `/sdcard` |
| CPU affinity | `taskset 60` |
| Threads | `--num_threads=1` |
| Warmup runs | 30 |
| Measurement runs | 40 |
| Latency label | `avg` inference time from benchmark output, converted to **milliseconds** |

### Kernel sampling (prior distribution)

Configs are sampled with nn-Meter prior samplers (same distributions as nn-Meter predictor build):

| Kernel | Requested samples | Profiled samples |
|--------|-------------------|------------------|
| Conv-BN-ReLU | 350 | 349 |
| FC | 200 | 200 |
| AvgPool | 100 | 100 |
| MaxPool | 100 | 100 |

## Results

### Sample counts and transfer source

| Kernel | Total | Train | Val | Similar pool device |
|--------|------:|------:|----:|---------------------|
| Conv-BN-ReLU | 349 | 279 | 70 | `xiaomi12tf21cpu_to_pixel6tf21cpu` |
| FC | 200 | 160 | 40 | `xiaomi12tf21cpu_to_pixel6tf21cpu` |
| AvgPool | 100 | 80 | 20 | `pixel5tf27cpu_to_xiaomi12tf27cpu` |
| MaxPool | 100 | 80 | 20 | `pixel5tf27cpu_to_xiaomi12tf27cpu` |

### Validation metrics (hold-out 20%)

| Kernel | RMSE (ms) | RMSPE (%) | Rel. error | Acc@5% | Acc@10% | Acc@15% |
|--------|----------:|----------:|-----------:|-------:|--------:|--------:|
| Conv-BN-ReLU | 207.26 | 16.33 | 1.18 | 51.4% | 77.1% | 85.7% |
| FC | 0.034 | 3.82 | 0.039 | 87.5% | 95.0% | 100% |
| AvgPool | 0.075 | 21.07 | 0.25 | 50.0% | 70.0% | 75.0% |
| MaxPool | 0.063 | 23.02 | 0.22 | 35.0% | 65.0% | 65.0% |

Full JSON report: `training_report.json`

## Generated artifacts

```
pixel9_tflite_cpu_20260608_194312/
├── datasets/
│   ├── Data_conv-bn-relu_litepred.csv
│   ├── Data_fc_litepred.csv
│   ├── Data_avgpool_litepred.csv
│   └── Data_maxpool_litepred.csv
├── predictors/
│   ├── conv-bn-relu.pth
│   ├── fc.pth
│   ├── avgpool.pth
│   └── maxpool.pth
├── kernels/                  # generated TFLite micro-benchmarks
├── training_report.json
└── profile_errors_*.json     # empty on this run (0 failures)
```

### Input features per kernel (LitePred MLP)

| Kernel | Features |
|--------|----------|
| Conv-BN-ReLU | HW, CIN, COUT, KERNEL_SIZE, STRIDES, FLOPS (M), PARAMS (M) |
| FC | CIN, COUT, FLOPS (M), PARAMS (M) |
| AvgPool / MaxPool | HW, CIN, COUT, KERNEL_SIZE, STRIDES |

## Assumptions and modifications

1. **LitePred-only training logic** — profiling (`profile_script`), similarity detection (`Detector`), and MLP training (`Trainer`) are from LitePred.

2. **TFLite model generation** — LitePred does not include a standalone kernel builder. Keras blocks from nn-Meter are used to synthesize kernels, then converted to `.tflite` before device profiling.

3. **Non-root ADB** — Pixel 9 has no `su`. `LitePred/profile_script/ADBConnect.py` was updated to run commands via `adb shell` without root.

4. **Remote path** — models are pushed to `/sdcard` (writable without root). The original `/data/local/tmp/nn_meter_models` path caused mmap permission errors for newly pushed files.

5. **SavedModel → TFLite** — nn-Meter `save_model` writes Keras SavedModel directories; the training script converts them to flat `.tflite` files before benchmarking.

6. **Transfer-learning pool** — three device pairs downloaded from HuggingFace into `LitePred/predictors/pool/`.

7. **Detector / Trainer fixes**
   - `detector.py`: dynamic input feature count per kernel; `torch.load(..., map_location='cpu')`
   - `trainer.py`: corrected FC input dimension from 5 → 4 (matches pretrained weights)

8. **Import isolation** — `train_pixel9_litepred.py` loads profile helpers via `importlib` to avoid shadowing Python's stdlib `profile` module.

9. **Latency units** — benchmark tool reports microseconds; LitePred CSV stores milliseconds (via `bench_utils.fetech_tf_bench_results`).

## Environment prerequisites

- ADB-connected Pixel 9 with USB debugging enabled
- On-device TFLite benchmark binary at `/data/local/tmp/benchmark_model`
- `energy-estimation` conda environment with: `torch`, `torchmetrics`, `tensorflow`, `pandas`, `scikit-learn`, `scipy`, `huggingface_hub`
