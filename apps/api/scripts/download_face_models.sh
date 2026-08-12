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

cat <<EOF
OpenCV fallback face models are ready in ${MODEL_DIR}.

For the stricter IDENTITY_ENGINE=scrfd_adaface stack, place these files in the
same directory:

  ${MODEL_DIR}/scrfd_2.5g_kps.onnx
  ${MODEL_DIR}/adaface_ir50_ms1mv2.onnx

Optional hardening models:

  ${MODEL_DIR}/minifasnet_antispoof.onnx
  ${MODEL_DIR}/6drepnet_300w_lp.onnx

The SCRFD project publishes KPS detector downloads from its official model zoo,
and AdaFace publishes R50 pretrained weights from its official repository. Review
model licensing with the client before government production use.
EOF
