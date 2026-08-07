#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_DIR="${ROOT_DIR}/models"

mkdir -p "${MODEL_DIR}"

download() {
  local url="$1"
  local target="$2"
  if [ -s "${target}" ]; then
    echo "exists: ${target}"
    return
  fi
  echo "downloading: ${url}"
  curl -fL --retry 3 --retry-delay 2 "${url}" -o "${target}.tmp"
  mv "${target}.tmp" "${target}"
}

download \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
  "${MODEL_DIR}/face_detection_yunet_2023mar.onnx"

download \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx" \
  "${MODEL_DIR}/face_recognition_sface_2021dec.onnx"

echo "Face models ready in ${MODEL_DIR}"
