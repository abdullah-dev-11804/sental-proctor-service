#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _shape_list(shape: object) -> list[object]:
    if isinstance(shape, (list, tuple)):
        result: list[object] = []
        for value in shape:
            if isinstance(value, (int, float)) and float(value).is_integer():
                result.append(int(value))
            else:
                result.append(value)
        return result
    return [shape]


def _materialize_shape(shape: list[object]) -> tuple[int, ...]:
    values: list[int] = []
    for index, item in enumerate(shape):
        if isinstance(item, int) and item > 0:
            values.append(item)
        elif index == 0:
            values.append(1)
        elif index == 1:
            values.append(3)
        else:
            values.append(224 if len(shape) >= 4 else 112)
    while len(values) < 4 and len(shape) >= 2:
        values.append(224 if len(values) > 1 else 1)
    return tuple(values if values else [1, 3, 224, 224])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an ONNX model's I/O layout.")
    parser.add_argument("model", type=Path, help="Path to an ONNX model")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required to inspect ONNX models.") from exc

    if not args.model.is_file():
        raise SystemExit(f"Model file not found: {args.model}")

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    inputs = []
    for info in session.get_inputs():
        inputs.append({
            "name": info.name,
            "shape": _shape_list(info.shape),
            "type": info.type,
        })
    outputs = []
    for info in session.get_outputs():
        outputs.append({
            "name": info.name,
            "shape": _shape_list(info.shape),
            "type": info.type,
        })

    payload = {
        "model": str(args.model),
        "inputs": inputs,
        "outputs": outputs,
    }

    if len(inputs) == 1:
        sample_shape = _materialize_shape(inputs[0]["shape"])
        sample = np.zeros(sample_shape, dtype=np.float32)
        try:
            model_outputs = session.run(None, {inputs[0]["name"]: sample})
            payload["test_input_shape"] = list(sample_shape)
            payload["test_output_shapes"] = [list(np.asarray(item).shape) for item in model_outputs]
        except Exception as exc:
            payload["test_error"] = str(exc)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
