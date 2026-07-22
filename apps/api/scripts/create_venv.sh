#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "No Python interpreter found. Install Python 3.11 or 3.12." >&2
  exit 1
fi

VERSION="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "$VERSION" in
  3.11|3.12)
    ;;
  *)
    echo "Unsupported Python $VERSION for this OpenCV service. Install Python 3.11 or 3.12, or use Docker." >&2
    exit 1
    ;;
esac

"$PYTHON_BIN" -m venv .venv
echo "Created .venv with $("$PYTHON_BIN" --version)"
