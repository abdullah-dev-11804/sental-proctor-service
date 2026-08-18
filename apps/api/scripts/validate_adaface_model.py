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
    if len(shape) >= 4:
        values = [shape[0], shape[1], shape[2], shape[3]]
    elif len(shape) == 2:
        values = [1, 3, shape[0], shape[1]]
    else:
        values = [1, 3, 112, 112]

    materialized: list[int] = []
    for index, item in enumerate(values):
        if isinstance(item, int) and item > 0:
            materialized.append(item)
        elif index == 0:
            materialized.append(1)
        elif index == 1:
            materialized.append(3)
        else:
            materialized.append(112)
    while len(materialized) < 4:
        materialized.append(112 if len(materialized) > 1 else 1)
    return tuple(materialized[:4])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an AdaFace ONNX model.")
    parser.add_argument("model", type=Path, help="Path to adaface_ir50_ms1mv2.onnx")
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required to inspect the AdaFace model.") from exc

    if not args.model.is_file():
        raise SystemExit(f"Model file not found: {args.model}")

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    input_shape = _shape_list(input_info.shape)
    output_shape = _shape_list(output_info.shape)
    test_shape = _materialize_shape(input_shape)

    sample = np.zeros(test_shape, dtype=np.float32)
    output = session.run(None, {input_info.name: sample})[0]
    embedding = np.asarray(output[0], dtype=np.float32).reshape(1, -1)
    norm = float(np.linalg.norm(embedding))
    normalized_norm = float(np.linalg.norm(embedding / norm)) if norm > 1e-6 else 0.0

    print(json.dumps({
        "model": str(args.model),
        "input_name": input_info.name,
        "input_shape": input_shape,
        "output_name": output_info.name,
        "output_shape": output_shape,
        "test_input_shape": list(test_shape),
        "embedding_norm": norm,
        "embedding_norm_after_normalize": normalized_norm,
    }, indent=2))


if __name__ == "__main__":
    main()
