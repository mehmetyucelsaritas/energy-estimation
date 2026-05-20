import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults
from nn_meter.builder.backends import BaseBackend, BaseParser, BaseProfiler

try:
    from .onnx_runner_compat import prepare_model_for_runner
except ImportError:
    from onnx_runner_compat import prepare_model_for_runner

logging = logging.getLogger("nn-Meter")

TRACE_PATH = "/data/misc/perfetto-traces/nnmeter.perfetto-trace"
CONFIG_PATH = "/data/misc/perfetto-configs/nnmeter_config.pbtxt"


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class Pixel9IDPProfiler(BaseProfiler):
    def __init__(
        self,
        serial: str,
        device_dir: str,
        remote_model_dir: str,
        runner_binary_path: str,
        runner_lib_ort_path: str,
        runner_lib_cpp_path: str,
        provider: str = "cpu",
        cpu_cluster: str = "all",
        warmup_s: int = 1,
        silence_s: int = 0,
        measurement_s: int = 3,
        cooldown_s: int = 0,
        perfetto_buffer_kb: int = 262144,
        perfetto_battery_poll_ms: int = 250,
        trace_processor_path: str = "IDP/trace_processor",
        power_rails: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        self.serial = serial or ""
        self.device_dir = device_dir
        self.remote_model_dir = remote_model_dir
        self.runner_binary_path = Path(runner_binary_path)
        self.runner_lib_ort_path = Path(runner_lib_ort_path)
        self.runner_lib_cpp_path = Path(runner_lib_cpp_path)
        self.provider = provider
        self.cpu_cluster = cpu_cluster
        self.warmup_s = int(warmup_s)
        self.silence_s = int(silence_s)
        self.measurement_s = int(measurement_s)
        self.cooldown_s = int(cooldown_s)
        self.perfetto_buffer_kb = int(perfetto_buffer_kb)
        self.perfetto_battery_poll_ms = int(perfetto_battery_poll_ms)
        self.trace_processor_path = Path(trace_processor_path)
        self.power_rails = power_rails or []
        self.verbose = _truthy(verbose)
        self._device_ready = False
        self._perfetto_pid: Optional[str] = None

    def _adb_base(self) -> List[str]:
        cmd = ["adb"]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def _adb(self, args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._adb_base() + list(args),
            check=check,
            capture_output=True,
            text=True,
        )

    def _adb_shell(self, shell_cmd: str, check: bool = True) -> str:
        res = self._adb(["shell", shell_cmd], check=check)
        return res.stdout

    def _generate_perfetto_config(self, duration_ms: int) -> str:
        return f"""buffers: {{
  size_kb: {self.perfetto_buffer_kb}
}}
data_sources: {{
  config {{
    name: "android.power"
    android_power_config {{
      collect_power_rails: true
      battery_poll_ms: {self.perfetto_battery_poll_ms}
      battery_counters: BATTERY_COUNTER_CAPACITY_PERCENT
      battery_counters: BATTERY_COUNTER_CHARGE
      battery_counters: BATTERY_COUNTER_CURRENT
      battery_counters: BATTERY_COUNTER_VOLTAGE
    }}
  }}
}}
duration_ms: {int(duration_ms)}
"""

    def _start_perfetto(self, duration_ms: int):
        cfg = self._generate_perfetto_config(duration_ms)
        with tempfile.NamedTemporaryFile("w", suffix=".pbtxt", delete=False) as tmp:
            tmp.write(cfg)
            tmp_path = tmp.name
        try:
            self._adb(["push", tmp_path, CONFIG_PATH], check=True)
        finally:
            os.unlink(tmp_path)

        self._adb_shell(f"rm -f {TRACE_PATH}", check=False)
        self._adb_shell("kill $(pgrep -x perfetto) 2>/dev/null || true", check=False)
        time.sleep(0.5)

        out = self._adb_shell(
            f"perfetto --txt --config {CONFIG_PATH} --out {TRACE_PATH} --background",
            check=False,
        )
        self._perfetto_pid = None
        for token in out.strip().split():
            if token.isdigit():
                self._perfetto_pid = token
                break
        if not self._perfetto_pid:
            fallback = self._adb_shell("pgrep -x perfetto", check=False).strip().splitlines()
            if fallback:
                last = fallback[-1].strip()
                if last.isdigit():
                    self._perfetto_pid = last
        time.sleep(1.5)

    def _stop_perfetto(self):
        time.sleep(1.0)
        if self._perfetto_pid:
            self._adb_shell(f"kill -INT {self._perfetto_pid} 2>/dev/null || true", check=False)
        else:
            self._adb_shell("kill -INT $(pgrep -x perfetto) 2>/dev/null || true", check=False)

        for _ in range(20):
            alive = self._adb_shell("pgrep -x perfetto", check=False).strip()
            if not alive:
                break
            time.sleep(1)
        else:
            self._adb_shell("kill -9 $(pgrep -x perfetto) 2>/dev/null || true", check=False)
            time.sleep(1)
        self._perfetto_pid = None

    def _pull_trace(self) -> Path:
        out_dir = Path(tempfile.gettempdir()) / "nnmeter_perfetto_traces"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"trace_{int(time.time() * 1000)}.perfetto-trace"
        self._adb(["pull", TRACE_PATH, str(out_path)], check=True)
        return out_path

    def _trace_processor_cmd(self) -> Optional[List[str]]:
        tp = self.trace_processor_path
        if not tp.is_absolute():
            repo_root = Path(__file__).resolve().parents[2]
            tp = repo_root / tp
        if tp.exists():
            return [sys.executable, str(tp)]
        shell_tp = shutil.which("trace_processor_shell")
        if shell_tp:
            return [shell_tp]
        return None

    def _run_trace_query(self, trace_path: Path, query: str) -> Optional[pd.DataFrame]:
        cmd = self._trace_processor_cmd()
        if not cmd:
            return None
        proc = subprocess.run(
            cmd + ["-Q", query, str(trace_path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        if proc.returncode != 0:
            return None
        csv_lines = [
            line for line in proc.stdout.splitlines()
            if not line.startswith("[") and not line.startswith("column")
        ]
        csv_text = "\n".join(csv_lines).strip()
        if not csv_text:
            return None
        try:
            return pd.read_csv(io.StringIO(csv_text))
        except Exception:
            return None

    @staticmethod
    def _compute_window_energy(
        power_df: pd.DataFrame,
        start_ms: float,
        end_ms: float,
        rails: List[str],
    ) -> Tuple[float, float]:
        if power_df is None or power_df.empty or end_ms <= start_ms:
            return 0.0, 0.0
        df = power_df.copy()
        df.columns = [str(c).strip().strip('"') for c in df.columns]
        for col in ("time_ms", "energy_uws"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["time_ms", "energy_uws", "rail"])
        if rails:
            mask = df["rail"].apply(lambda r: any(sub in str(r) for sub in rails))
            df = df[mask]
        if df.empty:
            return 0.0, 0.0

        total_energy_uws = 0.0
        for _, rdf in df.groupby("rail"):
            rdf = rdf.sort_values("time_ms").drop_duplicates(subset=["time_ms"], keep="last")
            if len(rdf) < 2:
                continue
            times = rdf["time_ms"].to_numpy(dtype=float)
            vals = rdf["energy_uws"].to_numpy(dtype=float)
            min_t, max_t = times[0], times[-1]
            if start_ms < min_t or end_ms > max_t:
                overlap = rdf[(rdf["time_ms"] >= start_ms - 2000) & (rdf["time_ms"] <= end_ms + 2000)]
                if len(overlap) < 2:
                    continue
                times = overlap["time_ms"].to_numpy(dtype=float)
                vals = overlap["energy_uws"].to_numpy(dtype=float)
            start_val = float(np.interp(start_ms, times, vals))
            end_val = float(np.interp(end_ms, times, vals))
            total_energy_uws += max(end_val - start_val, 0.0)

        energy_j = total_energy_uws / 1e6
        duration_s = max((end_ms - start_ms) / 1000.0, 1e-12)
        return energy_j, energy_j / duration_s

    def _ensure_device_ready(self):
        if self._device_ready:
            return
        self._adb(["start-server"], check=False)
        self._adb_shell(f"mkdir -p {self.device_dir} {self.remote_model_dir}", check=False)

        binary_dst = f"{self.device_dir}/{self.runner_binary_path.name}"
        lib_ort_dst = f"{self.device_dir}/{self.runner_lib_ort_path.name}"
        lib_cpp_dst = f"{self.device_dir}/{self.runner_lib_cpp_path.name}"

        self._adb(["push", str(self.runner_binary_path), binary_dst], check=True)
        self._adb(["push", str(self.runner_lib_ort_path), lib_ort_dst], check=True)
        self._adb(["push", str(self.runner_lib_cpp_path), lib_cpp_dst], check=True)
        self._adb_shell(f"chmod +x {binary_dst}", check=False)
        self._device_ready = True

    @staticmethod
    def _parse_runner_output(stdout: str) -> Optional[Dict[str, float]]:
        out: Dict[str, float] = {}

        m = re.search(r"Benchmark inferences:\s*(\d+)", stdout)
        if m:
            out["num_inferences"] = float(m.group(1))
        m = re.search(r"Average per inference:\s*([0-9.]+)", stdout)
        if m:
            out["usperinf"] = float(m.group(1)) * 1e6
        m = re.search(r"MEAS_START_BOOT_MS:\s*(\d+)", stdout)
        if m:
            out["meas_start_boot_ms"] = float(m.group(1))
        m = re.search(r"MEAS_END_BOOT_MS:\s*(\d+)", stdout)
        if m:
            out["meas_end_boot_ms"] = float(m.group(1))

        if out.get("num_inferences", 0.0) <= 0:
            return None
        return out

    def profile(self, graph_path, input_shape=None, **kwargs):
        _ = kwargs
        self._ensure_device_ready()

        local_model = Path(graph_path)
        runner_model_path, _compat_cache = prepare_model_for_runner(
            str(local_model),
            input_shape=input_shape,
        )
        runner_model = Path(runner_model_path)
        model_name = runner_model.name
        remote_model = f"{self.remote_model_dir}/{model_name}"
        self._adb(["push", str(runner_model), remote_model], check=True)

        nnapi_flag = "--nnapi" if self.provider == "nnapi" else ""
        cluster_flag = (
            f"--cpu-cluster {self.cpu_cluster}" if self.cpu_cluster and self.cpu_cluster != "all" else ""
        )
        runner_cmd = (
            f"cd {self.device_dir} && LD_LIBRARY_PATH=. ./run_onnx_arm64 "
            f"models/{model_name} {nnapi_flag} {cluster_flag} "
            f"--warmup {self.warmup_s} --silence {self.silence_s} --benchmark {self.measurement_s}"
        ).strip()

        duration_s = self.warmup_s + self.silence_s + self.measurement_s
        perfetto_duration_ms = int((duration_s + 10) * 1000)
        self._start_perfetto(perfetto_duration_ms)
        stdout = self._adb_shell(runner_cmd, check=False)
        self._stop_perfetto()
        trace_path = self._pull_trace()

        parsed = self._parse_runner_output(stdout)
        if parsed is None:
            raise RuntimeError(f"Failed to parse IDP runner output:\n{stdout[:500]}")

        usperinf = float(parsed["usperinf"])
        n_inf = float(parsed["num_inferences"])
        start_ms = float(parsed.get("meas_start_boot_ms", 0.0))
        end_ms = float(parsed.get("meas_end_boot_ms", 0.0))

        power_query = """
SELECT
  ts / 1000000 as time_ms,
  counter_track.name as rail,
  counter.value as energy_uws
FROM counter
JOIN counter_track ON counter.track_id = counter_track.id
WHERE counter_track.name LIKE 'power.rails.%'
ORDER BY ts
"""
        power_df = self._run_trace_query(trace_path, power_query)
        energy_j, avg_power_w = self._compute_window_energy(
            power_df if power_df is not None else pd.DataFrame(),
            start_ms,
            end_ms,
            self.power_rails,
        )
        mjperinf = (energy_j / n_inf) * 1000.0 if n_inf > 0 else 0.0

        if self.cooldown_s > 0:
            time.sleep(self.cooldown_s)

        payload = {
            "avg_ms": usperinf / 1000.0,
            "std_ms": 0.0,
            "energy_j": energy_j,
            "mjperinf": mjperinf,
            "num_inferences": n_inf,
            "meas_start_boot_ms": start_ms,
            "meas_end_boot_ms": end_ms,
        }
        if self.verbose:
            payload["runner_stdout_head"] = stdout[:300]
            payload["avg_power_w"] = avg_power_w
        return json.dumps(payload)


class Pixel9IDPParser(BaseParser):
    def __init__(self):
        self.latency = Latency()
        self.energy_mj_per_inf = 0.0

    def parse(self, content):
        data = json.loads(content)
        self.latency = Latency(float(data.get("avg_ms", 0.0)), float(data.get("std_ms", 0.0)))
        self.energy_mj_per_inf = float(data.get("mjperinf", 0.0))
        return self

    @property
    def results(self):
        return ProfiledResults({"latency": self.latency, "energy": self.energy_mj_per_inf})


class Pixel9CPUIDPBackend(BaseBackend):
    parser_class = Pixel9IDPParser
    profiler_class = Pixel9IDPProfiler

    def update_configs(self):
        super().update_configs()
        cfg = self.configs or {}

        repo_root = Path(__file__).resolve().parents[2]

        def _resolve_path(value: str) -> str:
            p = Path(value)
            return str(p if p.is_absolute() else (repo_root / p))

        self.profiler_kwargs.update(
            {
                "serial": cfg.get("DEVICE_SERIAL", ""),
                "device_dir": cfg.get("DEVICE_DIR", "/data/local/tmp/onnx_benchmark"),
                "remote_model_dir": cfg.get("REMOTE_MODEL_DIR", "/data/local/tmp/onnx_benchmark/models"),
                "runner_binary_path": _resolve_path(
                    cfg.get("RUNNER_BINARY_PATH", "IDP/device_runner/run_onnx_arm64")
                ),
                "runner_lib_ort_path": _resolve_path(
                    cfg.get(
                        "RUNNER_LIB_ORT_PATH",
                        "IDP/device_runner/ort_android/jni/arm64-v8a/libonnxruntime.so",
                    )
                ),
                "runner_lib_cpp_path": _resolve_path(
                    cfg.get(
                        "RUNNER_LIB_CPP_PATH",
                        "IDP/device_runner/ort_android/jni/arm64-v8a/libc++_shared.so",
                    )
                ),
                "provider": cfg.get("PROVIDER", "cpu"),
                "cpu_cluster": cfg.get("CPU_CLUSTER", "all"),
                "warmup_s": cfg.get("WARMUP_S", 1),
                "silence_s": cfg.get("SILENCE_S", 0),
                "measurement_s": cfg.get("MEASUREMENT_S", 3),
                "cooldown_s": cfg.get("COOLDOWN_S", 0),
                "perfetto_buffer_kb": cfg.get("PERFETTO_BUFFER_KB", 262144),
                "perfetto_battery_poll_ms": cfg.get("PERFETTO_BATTERY_POLL_MS", 250),
                "trace_processor_path": _resolve_path(cfg.get("TRACE_PROCESSOR_PATH", "IDP/trace_processor")),
                "power_rails": cfg.get("POWER_RAILS", []),
                "verbose": cfg.get("VERBOSE", False),
            }
        )

    def convert_model(self, model_path, save_path, input_shape=None):
        _ = input_shape
        src = Path(model_path)
        if src.suffix.lower() != ".onnx":
            raise ValueError(
                f"pixel9_cpu_idp expects ONNX model files (.onnx), got: {src}"
            )
        os.makedirs(save_path, exist_ok=True)
        dst = Path(save_path) / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return str(dst)

    def test_connection(self):
        serial = self.configs.get("DEVICE_SERIAL", "")
        cmd = ["adb"]
        if serial:
            cmd += ["-s", serial]
        cmd += ["shell", "echo hello backend !"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(res.stderr.strip() or "adb connection failed")
        logging.keyinfo(res.stdout.strip() or "hello backend !")
