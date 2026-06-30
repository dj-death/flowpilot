#!/usr/bin/env bash
# selfdrive/modeld/models/build_pickle.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ONNX="selfdrive/assets/models/driving/driving_supercombo.onnx"
OUT="selfdrive/modeld/models/driving_tinygrad.pkl"
GPU=1 python3 -m selfdrive.modeld.compile_modeld \
  --model-size 512x256 \
  --camera-resolutions 1920x1080 \
  --onnx "$ONNX" \
  --output "$OUT" \
  --frame-skip 4
