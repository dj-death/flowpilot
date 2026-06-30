#!/usr/bin/env bash
# selfdrive/modeld/models/fetch_driving_supercombo.sh
set -euo pipefail
OUT_DIR="$(cd "$(dirname "$0")/../../assets/models/driving" && pwd)"
OUT="$OUT_DIR/driving_supercombo.onnx"
SHA="659727c4d4839adc4992a254409a54259a8756a743f2d567bf5fdc6579f8009b"
OID="$SHA"   # LFS oid == sha256
SIZE="63963136"  # adjust if upstream re-points; sha is the real gate
mkdir -p "$OUT_DIR"
# Resolve the LFS download href via the GitLab batch API
HREF=$(curl -s -X POST \
  -H "Accept: application/vnd.git-lfs+json" \
  -H "Content-Type: application/vnd.git-lfs+json" \
  "https://gitlab.com/commaai/openpilot-lfs.git/info/lfs/objects/batch" \
  -d "{\"operation\":\"download\",\"transfers\":[\"basic\"],\"objects\":[{\"oid\":\"$OID\",\"size\":$SIZE}]}" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['objects'][0]['actions']['download']['href'])")
curl -L "$HREF" -o "$OUT"
echo "$SHA  $OUT" | sha256sum -c -
