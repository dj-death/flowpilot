# driving_supercombo tinygrad modeld Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run upstream openpilot v0.11 `driving_supercombo` in flowpilot by adding an on-device Python `modeld` co-process (tinygrad runtime) that bridges to flowpilot over the existing cereal/ZMQ bus.

**Architecture:** Vendor upstream `modeld.py` + its tinygrad model stack into flowpilot, swap its VisionIPC frame source for flowpilot's ZMQ `roadCameraBuffer`/`wideRoadCameraBuffer`, stub two unavailable inputs, and publish `modelV2`/`cameraOdometry` on the same ports flowpilot's UI and Python planners already consume. The Java `ModelExecutorF3` + native `modelparsed.cc` model path is disabled so there is a single publisher on port 5100.

**Tech Stack:** Python 3 (Termux on Android), tinygrad (pinned `f9c8c697`), capnp/cereal over ZMQ, OpenCL (Adreno) via pyopencl, the upstream `driving_supercombo.onnx` (openpilot v0.11.0).

## Global Constraints

- Target model: `driving_supercombo.onnx`, openpilot v0.11.0, GitLab-LFS object `sha256:659727c4d4839adc4992a254409a54259a8756a743f2d567bf5fdc6579f8009b` (61 MB). NOT `big_driving_supercombo.onnx`.
- Model input geometry: `img`/`big_img` = `[1,12,128,256]` uint8 → model_w=256, model_h=128, n_frames=2, 6 channels/frame. `frame_skip = MODEL_RUN_FREQ/MODEL_CONTEXT_FREQ = 20/5 = 4`.
- Float inputs (`features_buffer [1,24,512]`, `desire_pulse [1,25,8]`, `traffic_convention [1,2]`, `action_t [1,2]`) and output (`outputs [1,2576]`) are **float16** in the graph; the tinygrad pickle handles host-side float32↔fp16 casting, so feed float32 from Python.
- Output slicing comes from the `output_slices` dict embedded in the onnx metadata (captured into the pickle). Do not hardcode offsets in Python — read from `metadata['output_slices']`.
- Recurrence: `features_buffer` is refilled from `outputs[output_slices['hidden_state']]` (slice 1064:1576) each frame. Handled inside the vendored stack — do not reimplement.
- Use **this repo's** `cereal` package (its `cereal/resources/services.yaml` ordering + `cereal/log.capnp`). Never import stock openpilot cereal. Keep the `MSGQ` env var **unset** so the Python side uses the ZMQ backend.
- No capnp / `Definitions.java` changes. No control-stack changes. flowpilot's v0.9.4 planners keep deriving control from `position`/`orientation`.
- Camera resolution on the wire: 1920×1080 YUV420 (`Camera.frameSize`). Warp/pickle must be compiled for `1920x1080`.
- modeld camera mapping: `img` (main/road) ← `roadCameraBuffer`; `big_img` (wide) ← `wideRoadCameraBuffer`.
- Stubbed inputs (no compatible flowpilot source): `is_rhd = False`; `lat_delay = 0.2` (drop `driverMonitoringState` and `liveDelay` from the SubMaster).
- Vendored stack lives under `selfdrive/modeld/` in flowpilot, package-imported as `openpilot.selfdrive.modeld.*` (matches the existing in-repo Python package layout used by `flowinitd.py`).

---

## Phase M1 — On-device tinygrad runtime + model asset (highest-risk spike, gates everything)

### Task 1: Fetch and verify `driving_supercombo.onnx`

**Files:**
- Create: `selfdrive/modeld/models/fetch_driving_supercombo.sh`
- Create: `selfdrive/assets/models/driving/` (asset dir for the new model)

**Interfaces:**
- Produces: `selfdrive/assets/models/driving/driving_supercombo.onnx` (61 MB, verified sha256).

- [ ] **Step 1: Write the fetch script** (GitHub serves these LFS objects via GitLab; the GitLab batch API is the reliable source).

```bash
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
```

- [ ] **Step 2: Run it and verify the checksum**

Run: `bash selfdrive/modeld/models/fetch_driving_supercombo.sh`
Expected: final line `…/driving_supercombo.onnx: OK`. If the `size` field is rejected, read the actual size from `git cat-file`/error JSON and update `SIZE`; the sha256 check is the real gate.

- [ ] **Step 3: Confirm the onnx I/O matches the spec** (guards against a re-pointed upstream model)

```bash
python3 - <<'PY'
import onnx
m = onnx.load("selfdrive/assets/models/driving/driving_supercombo.onnx")
ins = {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in m.graph.input}
outs = {o.name: [d.dim_value for d in o.type.tensor_type.shape.dim] for o in m.graph.output}
print("inputs", ins); print("outputs", outs)
assert set(ins) == {"img","big_img","features_buffer","desire_pulse","traffic_convention","action_t"}, ins
assert outs["outputs"] == [1,2576], outs
print("OK")
PY
```
Expected: prints the 6 inputs + `outputs [1,2576]`, then `OK`.

- [ ] **Step 4: Commit**

```bash
git add selfdrive/modeld/models/fetch_driving_supercombo.sh
git commit -m "modeld: add driving_supercombo.onnx fetch+verify script"
# the .onnx itself: decide LFS vs direct commit in Task 5; do not commit the 61MB blob yet
```

---

### Task 2: Pin a tinygrad that can load/run the new model

**Files:**
- Create: `scripts/install_tinygrad_pinned.sh`
- Modify: `scripts/setup_tinygrad` (point it at the pinned commit)

**Interfaces:**
- Produces: an importable `tinygrad` at commit `f9c8c697d6bfb76ba0bd79e15704ed376f0c45ec` in the Python env (`python3 -c "import tinygrad"` works; `OnnxRunner` and `TinyJit` importable).

- [ ] **Step 1: Write the pinned installer**

```bash
#!/usr/bin/env bash
# scripts/install_tinygrad_pinned.sh
set -euo pipefail
TG_COMMIT="f9c8c697d6bfb76ba0bd79e15704ed376f0c45ec"
WORK="${1:-$HOME/tinygrad_pinned}"
if [ ! -d "$WORK/.git" ]; then git clone https://github.com/tinygrad/tinygrad "$WORK"; fi
git -C "$WORK" fetch --all
git -C "$WORK" checkout "$TG_COMMIT"
pip install -e "$WORK"
```

- [ ] **Step 2: Install and smoke-test the imports modeld needs**

Run:
```bash
bash scripts/install_tinygrad_pinned.sh
python3 -c "from tinygrad.tensor import Tensor; from tinygrad.engine.jit import TinyJit; from tinygrad.nn.onnx import OnnxRunner; from tinygrad.device import Device; print('tinygrad ok', Device.DEFAULT)"
```
Expected: `tinygrad ok <BACKEND>` with no ImportError. (Vendored `tinygrad_repo` v0.7.0 stays for legacy thneed compile only; the new stack imports the pinned install.)

- [ ] **Step 3: Commit**

```bash
git add scripts/install_tinygrad_pinned.sh scripts/setup_tinygrad
git commit -m "modeld: pin tinygrad f9c8c697 for new driving model"
```

---

### Task 3: Repair the Termux Python/OpenCL environment on-device

**Files:**
- Modify: `scripts/setup_tinygrad`

**Interfaces:**
- Produces: a Termux env where `pyopencl` sees the Adreno GPU and tinygrad selects an OpenCL/GPU device.

This is a device spike. Success = tinygrad runs a trivial kernel on the phone GPU.

- [ ] **Step 1: On the phone (Termux), build pyopencl against the Android OpenCL lib**

Run (in Termux):
```bash
pkg install python clang ocl-icd opencl-headers
ln -sf /vendor/lib64/libOpenCL.so $PREFIX/lib/libOpenCL.so 2>/dev/null || true
CL_LIBNAME=libOpenCL.so pip install pyopencl
python3 -c "import pyopencl as cl; print([d.name for p in cl.get_platforms() for d in p.get_devices()])"
```
Expected: a non-empty device list naming the Adreno GPU. If empty, locate the vendor `libOpenCL.so` (`find /vendor /system -name 'libOpenCL*.so'`) and re-link.

- [ ] **Step 2: Verify tinygrad runs a kernel on the GPU**

Run (Termux):
```bash
GPU=1 python3 -c "from tinygrad import Tensor, Device; print(Device.DEFAULT); print((Tensor([1.,2.,3.])*2).numpy())"
```
Expected: prints the GPU backend name and `[2. 4. 6.]`. **Gate:** if tinygrad cannot run on the Adreno GPU at all, stop and escalate — the whole approach depends on it (this is risk #1 from the spec).

- [ ] **Step 3: Record the working env in the script and commit**

Fold the verified steps into `scripts/setup_tinygrad` (idempotent), then:
```bash
git add scripts/setup_tinygrad
git commit -m "modeld: repair Termux tinygrad+pyopencl setup for Adreno OpenCL"
```

---

### Task 4: Compile the tinygrad pickle for flowpilot's camera resolution

**Files:**
- Create: `selfdrive/modeld/models/build_pickle.sh`

**Interfaces:**
- Consumes: vendored `compile_modeld.py` (Task 6 vendors the stack; this task may run after Task 6, but the command is fixed here).
- Produces: `selfdrive/assets/models/driving/driving_tinygrad.pkl` containing `{'metadata', 'run_policy', (1920,1080): warp_jit}`.

> Ordering note: this task depends on the vendored stack from Task 6. If executing strictly in order, do Task 6 first, then return here. Kept in M1 because compilation is part of the runtime spike.

- [ ] **Step 1: Write the build wrapper** (geometry from Global Constraints: model 256×128, camera 1920×1080, frame-skip 4)

```bash
#!/usr/bin/env bash
# selfdrive/modeld/models/build_pickle.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ONNX="selfdrive/assets/models/driving/driving_supercombo.onnx"
OUT="selfdrive/assets/models/driving/driving_tinygrad.pkl"
GPU=1 python3 -m openpilot.selfdrive.modeld.compile_modeld \
  --model-size 256x128 \
  --camera-resolutions 1920x1080 \
  --onnx "$ONNX" \
  --output "$OUT" \
  --frame-skip 4
```

- [ ] **Step 2: Build on the phone GPU and verify the self-test passes**

Run (Termux): `bash selfdrive/modeld/models/build_pickle.sh`
Expected: prints `capture + replay`, `pickle round trip`, per-run enqueue/total timings for `run_policy` and the `1920x1080` warp, and finally `Saved JITs to …/driving_tinygrad.pkl (NN.NN MB)`. The built-in `compile_jit` asserts pickle-round-trip determinism — a clean finish means the pickle is valid on this device.

- [ ] **Step 3: Verify the pickle loads and exposes the expected keys**

```bash
python3 - <<'PY'
import pickle
d = pickle.load(open("selfdrive/assets/models/driving/driving_tinygrad.pkl","rb"))
print(sorted(k for k in d if isinstance(k,str)))
assert 'metadata' in d and 'run_policy' in d and (1920,1080) in d
md = d['metadata']
assert 'output_slices' in md and 'input_shapes' in md
print("input_shapes", md['input_shapes'])
print("output_slices keys", list(md['output_slices']))
PY
```
Expected: lists `['metadata','run_policy']`, the input shapes, and `output_slices` keys including `hidden_state`, `plan`, `lane_lines`, …

- [ ] **Step 4: Commit the build script** (the `.pkl` asset handled in Task 5)

```bash
git add selfdrive/modeld/models/build_pickle.sh
git commit -m "modeld: add driving_tinygrad.pkl build wrapper"
```

---

### Task 5: Standalone on-device inference benchmark (the M1 gate)

**Files:**
- Create: `selfdrive/modeld/bench_driving_supercombo.py`

**Interfaces:**
- Consumes: `driving_tinygrad.pkl`, vendored `make_input_queues`, `WARP_INPUTS`, `POLICY_INPUTS`.
- Produces: a measured inference rate (Hz) on the phone GPU with zeroed inputs.

- [ ] **Step 1: Write the benchmark** (mirrors `ModelState.run` without messaging)

```python
# selfdrive/modeld/bench_driving_supercombo.py
import pickle, time, numpy as np
from tinygrad.tensor import Tensor
from openpilot.selfdrive.modeld.compile_modeld import make_input_queues, WARP_INPUTS, POLICY_INPUTS
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

CAM_W, CAM_H = 1920, 1080
PKL = "selfdrive/assets/models/driving/driving_tinygrad.pkl"

def main(n=100):
    d = pickle.load(open(PKL, "rb"))
    md, run_policy, warp = d["metadata"], d["run_policy"], d[(CAM_W, CAM_H)]
    iq, npy = make_input_queues(md["input_shapes"], 4, device=None)
    yuv_size = get_nv12_info(CAM_W, CAM_H)[3]
    frame = Tensor(np.zeros(yuv_size, dtype=np.uint8)).realize()
    ts = []
    for i in range(n):
        npy["tfm"][:] = np.eye(3, dtype=np.float32)
        npy["big_tfm"][:] = np.eye(3, dtype=np.float32)
        t0 = time.perf_counter()
        warped = warp(**{k: iq[k] for k in WARP_INPUTS}, frame=frame, big_frame=frame)
        outs, = run_policy(**{k: iq[k] for k in POLICY_INPUTS if k in iq}, warped=warped)
        _ = outs.numpy()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts[10:])  # drop warmup
    print(f"mean {ts.mean()*1e3:.1f} ms  ->  {1/ts.mean():.1f} Hz  (p95 {np.percentile(ts,95)*1e3:.1f} ms)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on-device and record the rate**

Run (Termux): `GPU=1 python3 selfdrive/modeld/bench_driving_supercombo.py`
Expected: a line like `mean NN.N ms -> NN.N Hz`. **Gate:** record this number. The road model targets ~20 Hz; if the phone GPU is far below (<~10 Hz) the integration is still useful but flag it — this is the measured decision point the user accepted by choosing B.

- [ ] **Step 3: Decide asset delivery and commit**

If `driving_tinygrad.pkl` + `.onnx` are committed to the repo, add them via the same mechanism the existing `f3/supercombo.thneed` uses (plain blob in this repo — `.gitattributes` LFS is not configured). Otherwise document that `fetch_*`/`build_pickle.sh` run at install time. Commit the benchmark:
```bash
git add selfdrive/modeld/bench_driving_supercombo.py
git commit -m "modeld: add standalone on-device inference benchmark"
```

---

## Phase M2 — Vendor and adapt `modeld.py`

### Task 6: Vendor the upstream modeld stack into flowpilot

**Files:**
- Create: `selfdrive/modeld/parse_model_outputs.py`, `selfdrive/modeld/fill_model_msg.py`, `selfdrive/modeld/constants.py`, `selfdrive/modeld/helpers.py`, `selfdrive/modeld/compile_modeld.py`, `selfdrive/modeld/get_model_metadata.py`, `selfdrive/modeld/models/__init__.py`
- Create: `selfdrive/modeld/modeld_flowpilot.py` (the adapted entrypoint — leave upstream `modeld.py` semantics intact but in a new file so the diff is reviewable)
- Create needed `common/transformations/` helpers if absent: `model.py` (`get_warp_matrix`), `camera.py`, `orientation.py`

**Interfaces:**
- Produces: importable `openpilot.selfdrive.modeld.{parse_model_outputs,fill_model_msg,constants,helpers,compile_modeld}` and `openpilot.common.transformations.model.get_warp_matrix`.

- [ ] **Step 1: Copy the upstream files verbatim** from openpilot master `openpilot/selfdrive/modeld/` (the scratchpad copies in `…/scratchpad/op/` are the fetched originals). Place them at the paths above. Keep imports as `openpilot.…`.

- [ ] **Step 2: Resolve transitive imports** — for each `from openpilot.common.… import …` used by the copied files, confirm it exists in flowpilot; vendor the missing module (notably `common/transformations/model.py`, `camera.py`, `orientation.py`, `common/file_chunker.py`, `common/filter_simple.py`). Search first:

Run: `python3 -c "import openpilot.selfdrive.modeld.parse_model_outputs" 2>&1 | tail -3`
Iterate: each `ModuleNotFoundError` names the next file to vendor. Stop when the import is clean.

- [ ] **Step 3: Import smoke test for every vendored module**

```bash
python3 - <<'PY'
import importlib
for m in ["openpilot.selfdrive.modeld.constants",
          "openpilot.selfdrive.modeld.parse_model_outputs",
          "openpilot.selfdrive.modeld.fill_model_msg",
          "openpilot.selfdrive.modeld.helpers",
          "openpilot.selfdrive.modeld.compile_modeld",
          "openpilot.common.transformations.model"]:
    importlib.import_module(m); print("ok", m)
PY
```
Expected: `ok …` for each, no errors.

- [ ] **Step 4: Commit**

```bash
git add selfdrive/modeld common/transformations common/file_chunker.py common/filter_simple.py
git commit -m "modeld: vendor upstream v0.11 modeld tinygrad stack"
```

---

### Task 7: Frame source adapter — reconstruct the model NV12 buffer from a flowpilot `FrameBuffer`

**Files:**
- Create: `selfdrive/modeld/flowpilot_frames.py`
- Test: `selfdrive/modeld/test_flowpilot_frames.py`

**Interfaces:**
- Produces: `class FlowpilotBuf` with a `.data` (contiguous `bytes`/`np.uint8` NV12 in the stride layout `get_nv12_info(1920,1080)` expects) and `frame_id`, `timestamp_sof`, `timestamp_eof`; and `read_frame(evt_bytes, which) -> FlowpilotBuf`.
- Consumed by Task 9.

flowpilot publishes `FrameBuffer` (`cereal/log.capnp:2176`) with `image`, `frameWidth/Height`, `stride`, `yWidth/yHeight/yPixelStride`, `uvWidth/uvHeight/uvPixelStride`, `uOffset`, `vOffset`, `encoding`. The warp (`make_frame_prepare`) expects **NV12 semi-planar** (`uv_offset = stride*y_height`, UV interleaved). This adapter normalizes flowpilot's frame into that exact layout.

- [ ] **Step 1: Write the failing test** with a synthetic NV12 and a synthetic I420 frame, asserting both normalize to the warp's NV12 layout.

```python
# selfdrive/modeld/test_flowpilot_frames.py
import numpy as np
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.selfdrive.modeld.flowpilot_frames import to_model_nv12

def _synthetic(w, h, planar):
    y = (np.arange(w*h) % 256).astype(np.uint8)
    if planar:  # I420: U plane then V plane
        u = np.full(w*h//4, 7, np.uint8); v = np.full(w*h//4, 9, np.uint8)
        return y.tobytes()+u.tobytes()+v.tobytes(), w  # stride==w
    uv = np.empty(w*h//2, np.uint8); uv[0::2]=7; uv[1::2]=9  # NV12 UVUV
    return y.tobytes()+uv.tobytes(), w

def test_i420_and_nv12_normalize_equal():
    w, h = 1920, 1080
    stride, y_h, uv_h, size = get_nv12_info(w, h)
    nv12_bytes, _ = _synthetic(w, h, planar=False)
    i420_bytes, _ = _synthetic(w, h, planar=True)
    a = to_model_nv12(nv12_bytes, w, h, stride=w, uv_pixel_stride=2, u_off=w*h, v_off=w*h+1)
    b = to_model_nv12(i420_bytes, w, h, stride=w, uv_pixel_stride=1, u_off=w*h, v_off=w*h+w*h//4)
    assert len(a) == size and len(b) == size
    assert a == b  # both produce identical NV12 the warp expects
```

- [ ] **Step 2: Run it, verify it fails**

Run: `pytest selfdrive/modeld/test_flowpilot_frames.py -v`
Expected: FAIL (`to_model_nv12` not defined).

- [ ] **Step 3: Implement `flowpilot_frames.py`**

```python
# selfdrive/modeld/flowpilot_frames.py
import numpy as np
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

def to_model_nv12(image: bytes, w: int, h: int, stride: int,
                  uv_pixel_stride: int, u_off: int, v_off: int) -> bytes:
    """Normalize a flowpilot YUV420 frame to the NV12 (UVUV) layout get_nv12_info expects."""
    m_stride, m_y_h, m_uv_h, m_size = get_nv12_info(w, h)
    src = np.frombuffer(image, dtype=np.uint8)
    out = np.zeros(m_size, dtype=np.uint8)
    # Y plane (de-stride source -> model stride)
    y_src = src[:stride*h].reshape(h, stride)[:, :w]
    out[:m_stride*m_y_h].reshape(m_y_h, m_stride)[:h, :w] = y_src
    uv_off = m_stride * m_y_h
    if uv_pixel_stride == 2:  # already NV12 semi-planar
        u = src[u_off:u_off + (w*h//2)][0::2][:w*h//4]
        v = src[u_off:u_off + (w*h//2)][1::2][:w*h//4]
    else:  # I420 planar
        u = src[u_off:u_off + w*h//4]
        v = src[v_off:v_off + w*h//4]
    inter = np.empty(w*h//2, np.uint8); inter[0::2] = u; inter[1::2] = v
    out[uv_off:uv_off + w*h//2] = inter
    return out.tobytes()

class FlowpilotBuf:
    def __init__(self, data: bytes, frame_id: int, ts_sof: int, ts_eof: int):
        self.data = data
        self.frame_id, self.timestamp_sof, self.timestamp_eof = frame_id, ts_sof, ts_eof

def read_frame(fb, fd) -> FlowpilotBuf:
    """fb = capnp FrameBuffer reader, fd = matching FrameData reader (frameId/timestamps)."""
    data = to_model_nv12(bytes(fb.image), fb.frameWidth, fb.frameHeight, fb.stride,
                         fb.uvPixelStride, fb.uOffset, fb.vOffset)
    return FlowpilotBuf(data, fd.frameId, fd.timestampSof, fd.timestampEof)
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `pytest selfdrive/modeld/test_flowpilot_frames.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add selfdrive/modeld/flowpilot_frames.py selfdrive/modeld/test_flowpilot_frames.py
git commit -m "modeld: add flowpilot FrameBuffer -> model NV12 adapter"
```

---

### Task 8: Warp-matrix parity using flowpilot's intrinsics

**Files:**
- Create: `selfdrive/modeld/flowpilot_warp.py`
- Test: `selfdrive/modeld/test_flowpilot_warp.py`

**Interfaces:**
- Produces: `warp_matrices(rpy_calib, road_intrinsics, wide_intrinsics) -> (tfm, big_tfm)` (each `np.float32` 3×3) using upstream `get_warp_matrix`, bypassing the `DEVICE_CAMERAS` lookup that flowpilot doesn't populate.
- Consumed by Task 9.

- [ ] **Step 1: Write the failing test** — asserts shapes/dtypes and that `tfm`(road, bigmodel=False) ≠ `big_tfm`(wide, bigmodel=True), and that zero-rpy gives a finite, non-identity matrix.

```python
# selfdrive/modeld/test_flowpilot_warp.py
import numpy as np
from openpilot.selfdrive.modeld.flowpilot_warp import warp_matrices

ROAD = np.array([[910.,0,256.],[0,910.,322.],[0,0,1.]], np.float32)
WIDE = np.array([[455.,0,256.],[0,455.,152.],[0,0,1.]], np.float32)

def test_warp_matrices_basic():
    tfm, big = warp_matrices(np.zeros(3, np.float32), ROAD, WIDE)
    assert tfm.shape == (3,3) and big.shape == (3,3)
    assert tfm.dtype == np.float32 and big.dtype == np.float32
    assert np.isfinite(tfm).all() and np.isfinite(big).all()
    assert not np.allclose(tfm, big)
```

- [ ] **Step 2: Run it, verify it fails**

Run: `pytest selfdrive/modeld/test_flowpilot_warp.py -v`
Expected: FAIL (`warp_matrices` not defined).

- [ ] **Step 3: Implement `flowpilot_warp.py`**

```python
# selfdrive/modeld/flowpilot_warp.py
import numpy as np
from openpilot.common.transformations.model import get_warp_matrix

def warp_matrices(rpy_calib, road_intrinsics, wide_intrinsics):
    rpy = np.asarray(rpy_calib, dtype=np.float32)
    tfm = get_warp_matrix(rpy, np.asarray(road_intrinsics, np.float32), False).astype(np.float32)
    big_tfm = get_warp_matrix(rpy, np.asarray(wide_intrinsics, np.float32), True).astype(np.float32)
    return tfm, big_tfm
```

- [ ] **Step 4: Run the test, verify it passes**

Run: `pytest selfdrive/modeld/test_flowpilot_warp.py -v`
Expected: PASS.

- [ ] **Step 5: Source flowpilot's actual intrinsics** — read `common/java/ai.flow.common/transformations/Camera.java` for `cam_intrinsics` (road) and the wide intrinsics; record both as the constants the entrypoint (Task 9) passes in. Commit:

```bash
git add selfdrive/modeld/flowpilot_warp.py selfdrive/modeld/test_flowpilot_warp.py
git commit -m "modeld: warp matrices from flowpilot intrinsics (bypass DEVICE_CAMERAS)"
```

---

### Task 9: Adapt the modeld entrypoint to flowpilot frames + stubbed inputs

**Files:**
- Modify: `selfdrive/modeld/modeld_flowpilot.py` (the vendored copy from Task 6)

**Interfaces:**
- Consumes: `FlowpilotBuf`/`read_frame` (Task 7), `warp_matrices` (Task 8), `ModelState` (unchanged upstream class).
- Produces: a runnable `main()` that subscribes to flowpilot frames + state and publishes `modelV2`/`cameraOdometry`.

Replace the VisionIPC machinery (upstream `modeld.py:149-174, 210-241, 273-274`) and the input sourcing (`:245-253`) per these diffs. Keep `ModelState`, `get_action_from_model`, and the publish block intact.

- [ ] **Step 1: Replace the VisionIPC setup block** (`modeld.py:149-175`) with flowpilot frame subscribers:

```python
  # flowpilot frame subscribers (no VisionIPC)
  import zmq
  from cereal import messaging as fp_msg
  frame_sub = messaging.SubMaster(["roadCameraState", "wideRoadCameraState"])
  road_buf_sock = messaging.sub_sock("roadCameraBuffer", conflate=True)
  wide_buf_sock = messaging.sub_sock("wideRoadCameraBuffer", conflate=True)
  main_wide_camera = False
  use_extra_client = True
  CAM_W, CAM_H = 1920, 1080
  st = time.monotonic()
  model = ModelState(CAM_W, CAM_H, USBGPU)
```

- [ ] **Step 2: Replace the frame-receive loop head** (`modeld.py:210-241`) with a flowpilot frame pull that yields `buf_main`/`buf_extra` via `read_frame`:

```python
  from openpilot.selfdrive.modeld.flowpilot_frames import read_frame
  while True:
    rb = messaging.recv_one_or_none(road_buf_sock)
    wb = messaging.recv_one_or_none(wide_buf_sock)
    frame_sub.update(0)
    if rb is None or wb is None:
      continue
    buf_main  = read_frame(rb.roadCameraBuffer,     frame_sub["roadCameraState"])
    buf_extra = read_frame(wb.wideRoadCameraBuffer, frame_sub["wideRoadCameraState"])
    meta_main, meta_extra = buf_main, buf_extra   # FlowpilotBuf carries frame_id/timestamps
```

- [ ] **Step 3: Replace input sourcing** (`modeld.py:243-253`) — stub the two unavailable inputs and use flowpilot intrinsics for the warp:

```python
    sm.update(0)
    desire = DH.desire
    is_rhd = False                                   # flowpilot has no compatible driverMonitoringState
    frame_id = sm["roadCameraState"].frameId if sm.seen["roadCameraState"] else meta_main.frame_id
    v_ego = max(sm["carState"].vEgo, 0.)
    lat_delay = 0.2 + LAT_SMOOTH_SECONDS             # flowpilot has no liveDelay service
    if sm.updated["liveCalibration"]:
      from openpilot.selfdrive.modeld.flowpilot_warp import warp_matrices
      from openpilot.selfdrive.modeld.flowpilot_intrinsics import ROAD_INTRINSICS, WIDE_INTRINSICS
      rpy = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      model_transform_main, model_transform_extra = warp_matrices(rpy, ROAD_INTRINSICS, WIDE_INTRINSICS)
      live_calib_seen = True
```

- [ ] **Step 4: Trim the SubMaster** (`modeld.py:179`) to flowpilot-available services:

```python
  sm = SubMaster(["carState", "liveCalibration", "carControl"])
```
(`deviceState`/`roadCameraState`/`driverMonitoringState`/`liveDelay` removed; `roadCameraState` is covered by `frame_sub`.)

- [ ] **Step 5: Create `selfdrive/modeld/flowpilot_intrinsics.py`** holding `ROAD_INTRINSICS`/`WIDE_INTRINSICS` (the values read in Task 8 Step 5), so the entrypoint imports resolve.

- [ ] **Step 6: Import/parse smoke test** (no device needed — guards syntax + imports)

Run: `python3 -c "import ast; ast.parse(open('selfdrive/modeld/modeld_flowpilot.py').read()); print('parse ok')"`
Expected: `parse ok`. Then `python3 -c "import openpilot.selfdrive.modeld.modeld_flowpilot"` and fix any ImportError.

- [ ] **Step 7: Commit**

```bash
git add selfdrive/modeld/modeld_flowpilot.py selfdrive/modeld/flowpilot_intrinsics.py
git commit -m "modeld: bridge entrypoint to flowpilot frames + stub is_rhd/lat_delay"
```

---

### Task 10: Drop the unused `drivingModelData` publish

**Files:**
- Modify: `selfdrive/modeld/modeld_flowpilot.py`

**Interfaces:**
- Produces: a single-purpose publisher set `["modelV2", "cameraOdometry"]` (flowpilot has no `drivingModelData` subscriber).

- [ ] **Step 1: Edit the PubMaster + send block** — change `pm = PubMaster(["modelV2", "drivingModelData", "cameraOdometry"])` (`modeld.py:178`) to `pm = PubMaster(["modelV2", "cameraOdometry"])`, and delete the `drivingdata_send = …` / `fill_driving_model_data(...)` / `pm.send('drivingModelData', …)` lines (`modeld.py:292, 309, 312`). Keep `fill_model_msg` and `fill_pose_msg`.

- [ ] **Step 2: Parse smoke test**

Run: `python3 -c "import ast; ast.parse(open('selfdrive/modeld/modeld_flowpilot.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add selfdrive/modeld/modeld_flowpilot.py
git commit -m "modeld: drop unused drivingModelData publish"
```

---

### Task 11: `modelV2` parity test against the onnxruntime reference

**Files:**
- Create: `selfdrive/modeld/test_modelv2_parity.py`

**Interfaces:**
- Consumes: `driving_supercombo.onnx`, vendored `Parser`, `slice_outputs`, the `output_slices` metadata.
- Produces: confidence that the vendored parse path reproduces the reference output decode.

This validates the *parse* path (the part that fills `modelV2`) independent of tinygrad, by running the raw onnx in onnxruntime and parsing with the vendored `Parser`.

- [ ] **Step 1: Write the test** — feed zeroed fp16 inputs to onnxruntime, parse with the vendored Parser, assert the decoded dict has the expected keys/shapes.

```python
# selfdrive/modeld/test_modelv2_parity.py
import numpy as np, onnxruntime as ort, pickle
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.selfdrive.modeld.constants import ModelConstants

ONNX = "selfdrive/assets/models/driving/driving_supercombo.onnx"
PKL  = "selfdrive/assets/models/driving/driving_tinygrad.pkl"

def test_parse_shapes():
    md = pickle.load(open(PKL, "rb"))["metadata"]
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    feeds = {}
    for i in sess.get_inputs():
        dt = np.float16 if i.type == 'tensor(float16)' else np.uint8
        feeds[i.name] = np.zeros([d for d in i.shape], dtype=dt)
    out = sess.run(None, feeds)[0][0]                      # (2576,) fp16
    sliced = {k: out[np.newaxis, v] for k, v in md["output_slices"].items()}
    parsed = Parser().parse_outputs(sliced)
    assert parsed["plan"].shape[1:] == (ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
    assert "lane_lines" in parsed and "lead" in parsed
    print({k: np.asarray(v).shape for k, v in parsed.items()})
```

- [ ] **Step 2: Run it**

Run: `pytest selfdrive/modeld/test_modelv2_parity.py -v -s`
Expected: PASS, printing decoded shapes. (If onnxruntime rejects fp16, re-export with float32 I/O — but the pickle/tinygrad path used at runtime does not need this; this test is a parser check only.)

- [ ] **Step 3: Commit**

```bash
git add selfdrive/modeld/test_modelv2_parity.py
git commit -m "modeld: add modelV2 parser parity test vs onnxruntime"
```

---

## Phase M3 — On-device integration

### Task 12: Register modeld as a managed process and gate off the Java path

**Files:**
- Modify: `selfdrive/manager/process_config.py`
- Modify: `common/java/ai.flow.common/utils.java` (runner mode)
- Modify: `android/src/main/java/ai.flow.android/AndroidLauncher.java`

**Interfaces:**
- Produces: the Python `modeld_flowpilot` launched by the manager; `ModelExecutorF3` + native `modelparsed` not started (single publisher on 5100).

- [ ] **Step 1: Read** `selfdrive/manager/process_config.py` to learn the `ManagerProcess`/`PythonProcess` pattern and the existing `F3`/`modelparsed` gating (the agent noted `modelparsed` is enabled when `F3`).

- [ ] **Step 2: Add the modeld process** following the existing pattern, e.g.:

```python
  PythonProcess("modeld", "openpilot.selfdrive.modeld.modeld_flowpilot",
                enabled=(os.getenv("MODELD_TG") == "1")),
```
and gate the existing native `modelparsed` entry so it is disabled when `MODELD_TG=1`.

- [ ] **Step 3: Gate the Java executor** — in `AndroidLauncher.java`, when the new tinygrad mode is selected, do **not** instantiate `ModelExecutorF3` (mirror how the commented-out `EXTERNAL_TINYGRAD` case skips the Java executor). Add a `USE_MODEL_RUNNER` value or reuse `EXTERNAL_TINYGRAD`; set it in `utils.java`. Read both files first and follow the existing switch structure.

- [ ] **Step 4: Build the Android app to confirm no compile regressions**

Run: `./gradlew :android:assembleDebug` (per the repo's documented build; see memory `build-apk.md` for the JDK 11 / NDK 27 / build-tools 34 workarounds).
Expected: BUILD SUCCESSFUL.

- [ ] **Step 5: Commit**

```bash
git add selfdrive/manager/process_config.py common/java/ai.flow.common/utils.java android/src/main/java/ai.flow.android/AndroidLauncher.java
git commit -m "modeld: launch tinygrad modeld co-process; gate off Java F3 path"
```

---

### Task 13: End-to-end on-device bring-up

**Files:**
- Create: `selfdrive/modeld/check_modelv2_live.py` (a subscriber assertion helper)

**Interfaces:**
- Produces: evidence that `modelV2`/`cameraOdometry` are published live and consumed by the UI/planners.

- [ ] **Step 1: Write a live checker**

```python
# selfdrive/modeld/check_modelv2_live.py
import time
from cereal import messaging
sm = messaging.SubMaster(["modelV2", "cameraOdometry"])
t0 = time.monotonic(); n = 0
while time.monotonic() - t0 < 10:
    sm.update(100)
    if sm.updated["modelV2"]:
        n += 1
        md = sm["modelV2"]
        assert len(md.position.x) > 0, "empty position"
print(f"modelV2 msgs in 10s: {n}  ({n/10:.1f} Hz)")
assert n > 0, "no modelV2 received"
```

- [ ] **Step 2: Launch flowpilot with the tinygrad model on-device** and run the checker

Run (Termux, with cameras + calibration active): start flowpilot with `MODELD_TG=1`, then `python3 selfdrive/modeld/check_modelv2_live.py`.
Expected: `modelV2 msgs in 10s: N (… Hz)` with N>0 and non-empty `position`. Confirm the UI (`OnRoadScreen`) renders the path and the Python planners receive `modelV2` (no `modelV2` timeouts in logs).

- [ ] **Step 3: Measure frame throughput/latency** — log `model_execution_time` and the receive rate; compare against the standalone benchmark (Task 5) to isolate ZMQ-frame overhead (spec risk #2).

- [ ] **Step 4: Commit**

```bash
git add selfdrive/modeld/check_modelv2_live.py
git commit -m "modeld: add live modelV2 publish checker"
```

---

## Phase M4 — Hardening

### Task 14: Calibration registration + perf tuning

**Files:**
- Modify: `selfdrive/modeld/modeld_flowpilot.py`, `selfdrive/modeld/flowpilot_intrinsics.py`

**Interfaces:**
- Produces: outputs that visually register correctly against the road, at the best achievable rate.

- [ ] **Step 1: Validate registration** — with the car/replay running, confirm predicted `laneLines`/`position` overlay the actual lane in the UI. If mis-registered, the intrinsics (Task 8/9) or the NV12 normalization (Task 7) are off — re-check `ROAD_INTRINSICS`/`WIDE_INTRINSICS` against `Camera.java` and the warp `M_inv` convention.

- [ ] **Step 2: Throughput tuning** — if the live rate (Task 13) is materially below the standalone rate (Task 5), reduce frame-copy overhead: ensure `conflate=True` on the buffer socks (already set), avoid redundant `bytes(...)` copies in `read_frame`, and consider down-scaling only if necessary. Re-measure.

- [ ] **Step 3: Document the measured on-device rate** in the spec's status and commit any tuning:

```bash
git add selfdrive/modeld/modeld_flowpilot.py selfdrive/modeld/flowpilot_intrinsics.py
git commit -m "modeld: calibration registration + frame-path perf tuning"
```

---

## Self-Review (completed during authoring)

- **Spec coverage:** §2 model I/O → Tasks 1,4,11; §3 messaging/process → Tasks 12,13; §4 schema (no changes) → asserted by Task 13 consumers; §5 input reconciliation → Task 9 (stubs); §6 data flow → Tasks 7–10; §7 components → Tasks 1–12; §8 milestones → phases M1–M4; §9 risks → Task 3 gate (runtime), Task 13 step 3 (throughput), Task 14 step 1 (parity). Non-goals (§10) respected: no capnp/control changes appear in any task.
- **Placeholder scan:** runtime-spike tasks (3, 5, 13) use concrete commands + measurable gates rather than fabricated unit tests — intentional for hardware bring-up, not placeholders.
- **Type consistency:** `to_model_nv12`/`read_frame`/`FlowpilotBuf` (Task 7) consumed in Task 9; `warp_matrices` (Task 8) consumed in Task 9; `ROAD_INTRINSICS`/`WIDE_INTRINSICS` defined Task 8/9 used in Task 9/14; `output_slices`/`metadata` keys consistent across Tasks 4, 11.

## Open risk acknowledged in-plan

The single largest risk — tinygrad reaching a usable frame rate on the Adreno GPU — is isolated to **Task 3 (gate)** and **Task 5 (measured rate)** before any integration work. If those fail, stop and revisit (the user accepted this by choosing Approach B straight-to-device).
