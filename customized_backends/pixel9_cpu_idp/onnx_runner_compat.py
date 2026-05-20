"""Prepare ONNX models for IDP run_onnx_arm64 runner limitations.

The device runner only feeds up to two inputs and duplicates the first input
shape for every extra input. Fusion-rule block models often need 2-3 inputs
with different static shapes, so we fold extra graph inputs into initializers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

_ShapeArg = Optional[Union[Sequence[int], Sequence[Sequence[int]]]]


def _shape_from_value_info(value_info) -> List[int]:
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_value <= 0:
            raise ValueError(f"Dynamic/unknown dim unsupported for runner compat: {value_info.name}")
        dims.append(int(dim.dim_value))
    return dims


def _numpy_dtype(elem_type: int) -> np.dtype:
    mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
        TensorProto.UINT8: np.uint8,
        TensorProto.INT8: np.int8,
    }
    if elem_type not in mapping:
        raise ValueError(f"Unsupported ONNX element type for runner compat: {elem_type}")
    return mapping[elem_type]


def _random_tensor(shape: Sequence[int], elem_type: int) -> np.ndarray:
    dtype = _numpy_dtype(elem_type)
    if np.issubdtype(dtype, np.floating):
        return (0.1 + 0.8 * np.random.rand(*shape)).astype(dtype)
    return np.random.randint(0, 4, size=shape, dtype=dtype)


def _needs_runner_compat(model: onnx.ModelProto) -> bool:
    graph = model.graph
    if len(graph.input) <= 1:
        return False
    shapes = [_shape_from_value_info(inp) for inp in graph.input]
    first = shapes[0]
    return any(shape != first for shape in shapes[1:]) or len(shapes) > 2


def prepare_model_for_runner(
    model_path: str,
    input_shape: _ShapeArg = None,
    cache_dir: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """Return a runner-compatible ONNX path and optional temp file to cleanup."""
    src = Path(model_path)
    model = onnx.load(str(src))
    if not _needs_runner_compat(model):
        return str(src), None

    graph = model.graph
    keep_input = graph.input[0]
    fold_inputs = list(graph.input[1:])

    for value_info in fold_inputs:
        shape = _shape_from_value_info(value_info)
        elem_type = value_info.type.tensor_type.elem_type
        data = _random_tensor(shape, elem_type)
        graph.initializer.append(numpy_helper.from_array(data, name=value_info.name))
        graph.input.remove(value_info)

    out_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "pixel9_runner_compat"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_runner_compat.onnx"
    onnx.save(model, str(dst))
    return str(dst), str(dst)
