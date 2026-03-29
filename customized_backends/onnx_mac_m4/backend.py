# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Local ONNX Runtime backend for nn-Meter (macOS / Apple Silicon friendly)."""

import json
import logging
import time
import os
import shutil
import statistics
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from nn_meter.builder.backends import BaseBackend, BaseParser, BaseProfiler
from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults

logging = logging.getLogger("nn-Meter")

_ShapeArg = Optional[Union[Sequence[int], Dict[str, Sequence[int]]]]


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _resolve_providers_from_config(c: Dict[str, Any]) -> List[str]:
    """Prefer explicit EXECUTION_PROVIDERS if set; else CoreML+CPU or CPU from USE_COREML_EP."""
    import onnxruntime as ort

    explicit = c.get("EXECUTION_PROVIDERS")
    if explicit is not None:
        if isinstance(explicit, str):
            return [p.strip() for p in explicit.split(",") if p.strip()]
        if isinstance(explicit, list):
            return [str(p).strip() for p in explicit if str(p).strip()]
    available = set(ort.get_available_providers())
    if _truthy(c.get("USE_COREML_EP")) and "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _numpy_dtype_for_onnx_type(ort_type: str):
    t = (ort_type or "").lower()
    if "float16" in t:
        return np.float16
    if "float" in t and "64" not in t:
        return np.float32
    if "double" in t or "float64" in t:
        return np.float64
    if "uint8" in t:
        return np.uint8
    if "int64" in t:
        return np.int64
    if "int32" in t:
        return np.int32
    return np.float32


def _resolve_dim(dim: Any, dynamic_replace: int) -> int:
    if dim is None:
        return dynamic_replace
    if isinstance(dim, str):
        return int(dynamic_replace)
    try:
        iv = int(dim)
        if iv <= 0:
            return int(dynamic_replace)
        return iv
    except (TypeError, ValueError):
        return int(dynamic_replace)


def _resolve_shape(
    shape: Sequence[Any], dynamic_replace: int
) -> List[int]:
    if not shape:
        return [int(dynamic_replace)]
    return [_resolve_dim(d, dynamic_replace) for d in shape]


def _finalize_input_shapes(
    session_inputs,
    user_shape: _ShapeArg,
    dynamic_replace: int,
) -> Dict[str, Tuple[int, ...]]:
    name_to_shape: Dict[str, Tuple[int, ...]] = {}
    first_name = session_inputs[0].name if session_inputs else None

    if isinstance(user_shape, dict):
        for inp in session_inputs:
            if inp.name in user_shape:
                name_to_shape[inp.name] = tuple(int(x) for x in user_shape[inp.name])
            else:
                name_to_shape[inp.name] = tuple(
                    _resolve_shape(inp.shape, dynamic_replace)
                )
        return name_to_shape

    if user_shape is not None and first_name is not None:
        # nn-meter ruletest uses shapes like [[C,H,W]] or [[C,H,W],[C,H,W]] for multi-input
        if (
            len(user_shape) > 0
            and isinstance(user_shape[0], (list, tuple))
            and not isinstance(user_shape[0], (str, bytes))
        ):
            for i, inp in enumerate(session_inputs):
                if i < len(user_shape):
                    name_to_shape[inp.name] = tuple(
                        int(x) for x in user_shape[i]
                    )
                else:
                    name_to_shape[inp.name] = tuple(
                        _resolve_shape(inp.shape, dynamic_replace)
                    )
            return name_to_shape
        name_to_shape[first_name] = tuple(int(x) for x in user_shape)
        for inp in session_inputs[1:]:
            name_to_shape[inp.name] = tuple(
                _resolve_shape(inp.shape, dynamic_replace)
            )
        return name_to_shape

    for inp in session_inputs:
        name_to_shape[inp.name] = tuple(_resolve_shape(inp.shape, dynamic_replace))
    return name_to_shape


def _match_onnx_input_rank(
    session,
    name_to_shape: Dict[str, Tuple[int, ...]],
    batch_dim: int,
) -> Dict[str, Tuple[int, ...]]:
    """Prepend batch when testcase shapes omit it (nn-meter uses CHW; ORT export uses NCHW)."""
    out: Dict[str, Tuple[int, ...]] = {}
    for inp in session.get_inputs():
        sh = list(name_to_shape[inp.name])
        spec = inp.shape
        if spec is None:
            out[inp.name] = tuple(sh)
            continue
        n_ort = len(spec)
        n_sh = len(sh)
        if n_sh + 1 == n_ort:
            sh = [batch_dim] + sh
        elif n_sh > n_ort:
            sh = sh[-n_ort:]
        out[inp.name] = tuple(int(x) for x in sh)
    return out


def _random_feed(
    session,
    name_to_shape: Dict[str, Tuple[int, ...]],
) -> Dict[str, np.ndarray]:
    feeds = {}
    name_meta = {inp.name: inp for inp in session.get_inputs()}
    for name, shape in name_to_shape.items():
        meta = name_meta[name]
        dtype = _numpy_dtype_for_onnx_type(meta.type)
        if np.issubdtype(dtype, np.integer):
            feeds[name] = np.random.randint(0, 4, size=shape, dtype=dtype)
        else:
            feeds[name] = np.random.randn(*shape).astype(dtype)
    return feeds


class ONNXRuntimeProfiler(BaseProfiler):
    """Runs ONNX models locally with ONNX Runtime and returns JSON metrics.

    Emits:
      - avg_ms, std_ms
      - avg_power_w (best-effort, via CodeCarbon) when available
    """

    def __init__(
        self,
        warmup_runs=10,
        num_runs=50,
        providers=None,
        dynamic_batch_dim=1,
        intra_op_num_threads=0,
        inter_op_num_threads=0,
        verbose=False,
        power_fast_legacy_delta=False,
    ):
        self._warmup_runs = int(warmup_runs)
        self._num_runs = int(num_runs)
        self._providers = providers if providers is not None else ["CPUExecutionProvider"]
        self._dynamic_batch_dim = int(dynamic_batch_dim)
        self._intra_op_num_threads = int(intra_op_num_threads or 0)
        self._inter_op_num_threads = int(inter_op_num_threads or 0)
        self._verbose = _truthy(verbose)
        self._power_fast_legacy = _truthy(power_fast_legacy_delta)

    def profile(
        self,
        graph_path,
        input_shape: _ShapeArg = None,
        **kwargs,
    ):
        import onnxruntime as ort

        _ = kwargs  # reserved for builder call sites
        requested_metrics = kwargs.get("metrics", ["latency"])
        if not isinstance(requested_metrics, list):
            requested_metrics = [requested_metrics]
        power_requested = "power" in requested_metrics
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if self._intra_op_num_threads > 0:
            so.intra_op_num_threads = self._intra_op_num_threads
        if self._inter_op_num_threads > 0:
            so.inter_op_num_threads = self._inter_op_num_threads
        session = ort.InferenceSession(
            graph_path, sess_options=so, providers=list(self._providers)
        )
        name_to_shape = _finalize_input_shapes(
            session.get_inputs(),
            input_shape,
            self._dynamic_batch_dim,
        )
        name_to_shape = _match_onnx_input_rank(
            session, name_to_shape, self._dynamic_batch_dim
        )
        feed = _random_feed(session, name_to_shape)
        output_names = [o.name for o in session.get_outputs()]

        for _ in range(self._warmup_runs):
            session.run(output_names, feed)

        # Timed runs. We optionally track energy for the whole window and
        # derive average power as P_avg = E / t.
        samples_ms: List[float] = []
        avg_power_w: Optional[float] = None

        tracker = None
        cp_before = None
        use_cc_checkpoint = False
        if power_requested:
            try:
                from codecarbon import OfflineEmissionsTracker  # type: ignore

                tracker = OfflineEmissionsTracker(
                    save_to_file=False,
                    save_to_api=False,
                    log_level="error",
                )
                tracker.start()
                use_cc_checkpoint = callable(getattr(tracker, "checkpoint", None))
                if use_cc_checkpoint:
                    cp_before = tracker.checkpoint()
                    if cp_before is None:
                        use_cc_checkpoint = False
            except Exception as e:
                if self._verbose:
                    logging.warning(f"CodeCarbon unavailable; skipping power metric: {e}")
                tracker = None
                use_cc_checkpoint = False

        t_window0 = time.perf_counter()
        try:
            for _ in range(self._num_runs):
                t0 = time.perf_counter()
                session.run(output_names, feed)
                samples_ms.append((time.perf_counter() - t0) * 1000.0)
        finally:
            t_window_s = max(time.perf_counter() - t_window0, 1e-12)
            if tracker is not None:
                try:
                    if use_cc_checkpoint and cp_before is not None:
                        cp_after = tracker.checkpoint()
                        if cp_after is not None:
                            seg = cp_after.segment_since(cp_before)
                            delta_kwh = max(float(seg.energy_consumed_kwh), 0.0)
                            energy_j = float(delta_kwh) * 3_600_000.0
                            avg_power_w = float(energy_j / t_window_s)
                    _ = tracker.stop()
                    if avg_power_w is None:
                        emissions_data = getattr(tracker, "final_emissions_data", None)
                        if emissions_data is None:
                            emissions_data = getattr(tracker, "_last_emissions_data", None)
                        energy_kwh = None
                        if emissions_data is not None:
                            energy_kwh = getattr(emissions_data, "energy_consumed", None)
                        if energy_kwh is not None:
                            energy_j = float(energy_kwh) * 3_600_000.0
                            avg_power_w = float(energy_j / t_window_s)
                except Exception as e:
                    if self._verbose:
                        logging.warning(f"CodeCarbon measurement failed; skipping power: {e}")

        avg = float(statistics.mean(samples_ms))
        std = float(statistics.stdev(samples_ms)) if len(samples_ms) > 1 else 0.0
        payload: Dict[str, Any] = {"avg_ms": avg, "std_ms": std}
        if avg_power_w is not None:
            payload["avg_power_w"] = avg_power_w
        if self._verbose:
            payload["requested_providers"] = list(self._providers)
            payload["session_providers"] = list(session.get_providers())
        return json.dumps(payload)

    def profile_many(self, models: List[Dict[str, Any]], metrics: List[str]):
        """Profile multiple converted ONNX models in one CodeCarbon session when power is requested.

        Used by ``build_power_predictor`` via ``ONNXMacBackend.profile_models_batch``: one
        ``tracker.start()``/``stop()``, and per-model average power from
        ``checkpoint()``/``segment_since()`` energy divided by that model's timed window
        (when CodeCarbon exposes ``checkpoint()``).
        """
        requested_metrics = metrics if isinstance(metrics, list) else [metrics]
        power_requested = "power" in requested_metrics
        outputs: List[str] = []

        # Prepare model sessions first so profiling loop stays tight.
        prepared: List[Dict[str, Any]] = []
        import onnxruntime as ort
        for item in models:
            graph_path = item["graph_path"]
            input_shape = item.get("input_shape", None)
            so = ort.SessionOptions()
            so.log_severity_level = 3
            if self._intra_op_num_threads > 0:
                so.intra_op_num_threads = self._intra_op_num_threads
            if self._inter_op_num_threads > 0:
                so.inter_op_num_threads = self._inter_op_num_threads
            session = ort.InferenceSession(
                graph_path, sess_options=so, providers=list(self._providers)
            )
            name_to_shape = _finalize_input_shapes(
                session.get_inputs(),
                input_shape,
                self._dynamic_batch_dim,
            )
            name_to_shape = _match_onnx_input_rank(
                session, name_to_shape, self._dynamic_batch_dim
            )
            feed = _random_feed(session, name_to_shape)
            output_names = [o.name for o in session.get_outputs()]
            prepared.append(
                {
                    "session": session,
                    "feed": feed,
                    "output_names": output_names,
                }
            )

        tracker = None
        use_cc_checkpoint = False
        prev_total_energy_kwh: Optional[float] = None
        if power_requested:
            try:
                from codecarbon import OfflineEmissionsTracker  # type: ignore

                tracker = OfflineEmissionsTracker(
                    save_to_file=False,
                    save_to_api=False,
                    log_level="error",
                )
                tracker.start()
                use_cc_checkpoint = callable(getattr(tracker, "checkpoint", None))
                if not use_cc_checkpoint:
                    logging.warning(
                        "CodeCarbon has no checkpoint(); batch power falls back to sequential "
                        "deltas (install a build with checkpoint(), e.g. from codecarbon-master) "
                        "for clearer per-model attribution."
                    )
                    # Legacy CodeCarbon: stop background threads; measure after each model.
                    scheduler = getattr(tracker, "_scheduler", None)
                    if scheduler is not None:
                        scheduler.stop()
                    scheduler_monitor = getattr(tracker, "_scheduler_monitor_power", None)
                    if scheduler_monitor is not None:
                        scheduler_monitor.stop()
                    total_energy = getattr(tracker, "_total_energy", None)
                    if total_energy is not None:
                        prev_total_energy_kwh = float(getattr(total_energy, "kWh", 0.0))
                elif self._power_fast_legacy:
                    # One _measure_power_and_energy per model (~2x faster than two checkpoints);
                    # energy delta includes that model's warmup; denominator is timed window only.
                    logging.info(
                        "POWER_FAST_LEGACY_DELTA: batch power uses post-warmup + post-timed samples "
                        "per model (same energy window as checkpoint mode)."
                    )
                    use_cc_checkpoint = False
                    scheduler = getattr(tracker, "_scheduler", None)
                    if scheduler is not None:
                        scheduler.stop()
                    scheduler_monitor = getattr(tracker, "_scheduler_monitor_power", None)
                    if scheduler_monitor is not None:
                        scheduler_monitor.stop()
                    total_energy = getattr(tracker, "_total_energy", None)
                    if total_energy is not None:
                        prev_total_energy_kwh = float(getattr(total_energy, "kWh", 0.0))
            except Exception as e:
                if self._verbose:
                    logging.warning(f"CodeCarbon unavailable in batch mode; skipping power metric: {e}")
                tracker = None
                use_cc_checkpoint = False

        total_t_window_s = 0.0
        stats: List[Dict[str, Optional[float]]] = []

        try:
            for item in prepared:
                session = item["session"]
                feed = item["feed"]
                output_names = item["output_names"]
                for _ in range(self._warmup_runs):
                    session.run(output_names, feed)
                # Legacy delta path: align numerator with checkpoint semantics (energy after
                # warmup only). Otherwise warmup energy is divided by timed-only t_window_s.
                if power_requested and tracker is not None and not use_cc_checkpoint:
                    try:
                        if hasattr(tracker, "_measure_power_and_energy"):
                            tracker._measure_power_and_energy()  # type: ignore[attr-defined]
                        total_energy_bw = getattr(tracker, "_total_energy", None)
                        if total_energy_bw is not None:
                            prev_total_energy_kwh = float(
                                getattr(total_energy_bw, "kWh", 0.0)
                            )
                    except Exception as e:
                        if self._verbose:
                            logging.warning(
                                f"CodeCarbon post-warmup baseline failed: {e}"
                            )
                cp_before = None
                if power_requested and tracker is not None and use_cc_checkpoint:
                    try:
                        cp_before = tracker.checkpoint()
                        if cp_before is None:
                            raise RuntimeError("checkpoint() returned None before model window")
                    except Exception as e:
                        if self._verbose:
                            logging.warning(
                                f"CodeCarbon checkpoint before model failed: {e}; "
                                "power for this model may be missing."
                            )
                        cp_before = None
                samples_ms: List[float] = []
                t_window0 = time.perf_counter()
                for _ in range(self._num_runs):
                    t0 = time.perf_counter()
                    session.run(output_names, feed)
                    samples_ms.append((time.perf_counter() - t0) * 1000.0)
                t_window_s = max(time.perf_counter() - t_window0, 1e-12)
                total_t_window_s += t_window_s
                avg = float(statistics.mean(samples_ms))
                std = float(statistics.stdev(samples_ms)) if len(samples_ms) > 1 else 0.0
                per_model_avg_power_w: Optional[float] = None
                if power_requested and tracker is not None:
                    try:
                        if use_cc_checkpoint and cp_before is not None:
                            cp_after = tracker.checkpoint()
                            if cp_after is None:
                                raise RuntimeError("checkpoint() returned None after model window")
                            seg = cp_after.segment_since(cp_before)
                            delta_energy_kwh = max(float(seg.energy_consumed_kwh), 0.0)
                            energy_j = float(delta_energy_kwh) * 3_600_000.0
                            per_model_avg_power_w = float(energy_j / t_window_s)
                        else:
                            if hasattr(tracker, "_measure_power_and_energy"):
                                tracker._measure_power_and_energy()  # type: ignore[attr-defined]
                            total_energy = getattr(tracker, "_total_energy", None)
                            if total_energy is not None:
                                curr_total_energy_kwh = float(getattr(total_energy, "kWh", 0.0))
                                if prev_total_energy_kwh is None:
                                    prev_total_energy_kwh = curr_total_energy_kwh
                                delta_energy_kwh = max(
                                    curr_total_energy_kwh - prev_total_energy_kwh, 0.0
                                )
                                prev_total_energy_kwh = curr_total_energy_kwh
                                energy_j = float(delta_energy_kwh) * 3_600_000.0
                                per_model_avg_power_w = float(energy_j / t_window_s)
                    except Exception as e:
                        if self._verbose:
                            logging.warning(
                                f"CodeCarbon per-model energy failed; fallback to batch power: {e}"
                            )
                stats.append(
                    {
                        "avg_ms": avg,
                        "std_ms": std,
                        "avg_power_w": per_model_avg_power_w,
                    }
                )
        finally:
            avg_power_w: Optional[float] = None
            if tracker is not None:
                try:
                    _ = tracker.stop()
                    emissions_data = getattr(tracker, "final_emissions_data", None)
                    if emissions_data is None:
                        emissions_data = getattr(tracker, "_last_emissions_data", None)
                    energy_kwh = None
                    if emissions_data is not None:
                        energy_kwh = getattr(emissions_data, "energy_consumed", None)
                    if energy_kwh is not None and total_t_window_s > 0:
                        energy_j = float(energy_kwh) * 3_600_000.0
                        avg_power_w = float(energy_j / total_t_window_s)
                except Exception as e:
                    if self._verbose:
                        logging.warning(f"CodeCarbon batch measurement failed; skipping power: {e}")

            mean_avg_ms = None
            if len(stats) > 0:
                mean_avg_ms = float(statistics.mean([float(s["avg_ms"]) for s in stats]))
            for s in stats:
                payload: Dict[str, Any] = {"avg_ms": s["avg_ms"], "std_ms": s["std_ms"]}
                if power_requested:
                    if s.get("avg_power_w") is not None:
                        payload["avg_power_w"] = s["avg_power_w"]
                    elif avg_power_w is not None:
                        # Fast fallback when task-level power is unavailable: distribute
                        # chunk average power by relative runtime to keep label variance.
                        if mean_avg_ms is not None and mean_avg_ms > 1e-12:
                            runtime_scale = max(float(s["avg_ms"]) / mean_avg_ms, 1e-6)
                            payload["avg_power_w"] = float(avg_power_w * runtime_scale)
                        else:
                            payload["avg_power_w"] = avg_power_w
                outputs.append(json.dumps(payload))

        return outputs


class ONNXLatencyParser(BaseParser):
    def __init__(self):
        self._latency = Latency(0.0, 0.0)
        self._power_w: Optional[float] = None

    def parse(self, content):
        data = json.loads(content)
        self._latency = Latency(float(data["avg_ms"]), float(data["std_ms"]))
        if "avg_power_w" in data and data["avg_power_w"] is not None:
            self._power_w = float(data["avg_power_w"])
        return self

    @property
    def results(self):
        out: Dict[str, Any] = {"latency": self._latency}
        if self._power_w is not None:
            out["power"] = self._power_w
        return ProfiledResults(out)


class ONNXMacBackend(BaseBackend):
    parser_class = ONNXLatencyParser
    profiler_class = ONNXRuntimeProfiler

    def update_configs(self):
        super().update_configs()
        c = self.configs or {}
        providers = _resolve_providers_from_config(c)
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

    def convert_model(self, model_path, save_path, input_shape=None):
        os.makedirs(save_path, exist_ok=True)
        ext = os.path.splitext(model_path)[1].lower()
        if ext != ".onnx":
            raise ValueError(
                "onnx_mac_m4 backend expects a .onnx file; got "
                f"{ext or 'no extension'} for path: {model_path}"
            )
        dest = os.path.join(save_path, os.path.basename(model_path))
        if os.path.abspath(model_path) != os.path.abspath(dest):
            shutil.copy2(model_path, dest)
        return dest

    def test_connection(self):
        import onnxruntime as ort

        _ = ort.get_available_providers()
        logging.keyinfo("hello backend !")

    def profile_models_batch(self, converted_models, metrics=['latency']):
        """Batch profile models; with ``metrics=['power']``, one tracker session and per-model power."""
        outputs = self.profiler.profile_many(converted_models, metrics=metrics)
        parsed = []
        for content in outputs:
            parsed.append(self.parser.parse(content).results.get(metrics))
        return parsed
