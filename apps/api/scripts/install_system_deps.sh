#!/usr/bin/env bash
set -euo pipefail

if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    curl \
    docker.io \
    docker-compose-plugin
elif command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y \
    python3.12 \
    python3.12-devel \
    python3-pip \
    ffmpeg \
    glib2 \
    mesa-libGL \
    curl \
    docker \
    docker-compose-plugin
else
  echo "Unsupported package manager. Install Python 3.11/3.12, ffmpeg, Docker, and OpenCV runtime libs manually." >&2
  exit 1
fi

echo "System dependencies installed."
