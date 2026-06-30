#!/usr/bin/env bash
# scripts/install_tinygrad_pinned.sh
set -euo pipefail
TG_COMMIT="f9c8c697d6bfb76ba0bd79e15704ed376f0c45ec"
WORK="${1:-$HOME/tinygrad_pinned}"
if [ ! -d "$WORK/.git" ]; then git clone https://github.com/tinygrad/tinygrad "$WORK"; fi
git -C "$WORK" fetch --all
git -C "$WORK" checkout "$TG_COMMIT"
pip install -e "$WORK"
