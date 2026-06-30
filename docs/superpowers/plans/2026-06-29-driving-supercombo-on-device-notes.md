# driving_supercombo modeld — on-device bring-up notes

Companion to the spec (`…specs/2026-06-29-driving-supercombo-tinygrad-modeld-design.md`) and plan (`…plans/2026-06-29-driving-supercombo-modeld.md`). All code-authoring tasks are committed on `feat/driving-supercombo-modeld`. Below is the review punch-list: what was fixed during authoring, and what still must be done/verified on the phone (none of it could be executed from the Windows dev session — no numpy/tinygrad locally, and Termux is required).

## How to run on-device (smoke path)
```bash
cd <flowpilot BASEDIR on phone>        # e.g. /sdcard/flowpilot
export PYTHONPATH=$PWD                  # flat imports: from common/selfdrive/cereal
bash scripts/install_tinygrad_pinned.sh         # tinygrad f9c8c697
bash scripts/setup_tinygrad                     # Termux pyopencl vs Adreno libOpenCL
bash selfdrive/modeld/models/fetch_driving_supercombo.sh   # onnx -> selfdrive/assets/models/driving/
bash selfdrive/modeld/models/build_pickle.sh               # -> selfdrive/modeld/models/driving_tinygrad.pkl
GPU=1 python3 selfdrive/modeld/bench_driving_supercombo.py # M1 GATE: measure Hz
# then full app with MODELD_TG=1 (selects tinygrad modeld; disables Java F3 + native modelparsed)
```

## Fixed during authoring (committed)
- Vendored upstream v0.11 modeld stack; merged `ModelConstants/Plan/Meta` into the existing `constants.py` without breaking the control stack's `T_IDXS`/`index_function` exports.
- Normalized all imports from `openpilot.*` to flowpilot's flat layout (`common/selfdrive/cereal/system`); dropped `msgq.visionipc` and `opendbc`; vendored `system/camerad/cameras/nv12_info.py`; removed `common.hardware.hw.Paths` use in `compile_modeld.py`.
- Added `get_warp_matrix_intrinsics` (explicit-intrinsics variant) and pointed `flowpilot_warp.py` at it.
- Adapted `fill_model_msg`/`modeld_flowpilot` to flowpilot's `ModelDataV2` schema: removed writes to absent fields (`action`, `meta.laneChange*`, `DisengagePredictions.gas/brakePressProbs`) and the `get_action_from_model` path.
- Fixed the pickle load path (`modeld_pkl_path()` → `selfdrive/modeld/models/driving_tinygrad.pkl`) across `build_pickle.sh`/`bench`/`parity-test`.
- Added `selfdrive/modeld/models/tg_input_devices.json` (read at startup by `get_tg_input_devices`).
- Replaced the upstream `from_blob` pointer-cache in `ModelState.run` with a per-frame device `Tensor` (flowpilot frames are fresh `bytes`, so the pointer cache would leak and dangle).

## Must verify / finish on-device (review findings, by severity)

### Gating risk
- **tinygrad on Adreno at a usable rate** — the M1 gate (`bench_driving_supercombo.py`). Everything depends on this. Record the Hz; road model targets ~20 Hz.

### Important (will break or mis-behave until addressed)
1. **`tg_input_devices.json` device strings** — `WARP_DEV`/`QUEUE_DEV` are placeholder `"GPU"`. Set to the actual tinygrad device for the phone (e.g. `"QCOM"` on comma-class hardware, or the correct OpenCL device name). Wrong values → load/run failure.
2. **`to_model_nv12` UV plane is not stride-aware** (`flowpilot_frames.py`). The Y plane de-strides correctly, but the U/V read (from `uOffset`/`vOffset`) and the model-NV12 UV write assume `stride == frameWidth` and a contiguous `w*h/2` UV region. If `FrameBuffer.stride > frameWidth` (padded), UV is misread/miswritten → corrupted chroma. Make UV use `uvWidth/uvHeight/stride` symmetrically with Y.
3. **`messaging.recv_one_or_none` / `messaging.sub_sock(..., conflate=True)`** — confirm both exist in flowpilot's `cereal.messaging`. If not, swap for the available equivalent (e.g. `sub_sock` + `receive(non_blocking=True)`).
4. **`DesireHelper.update(carState, latActive, lane_change_prob)`** — confirm flowpilot's `selfdrive/controls/lib/desire_helper.py` has this signature; crash if it differs.

### Minor / tuning
5. **`WIDE_INTRINSICS == ROAD_INTRINSICS`** (`flowpilot_intrinsics.py`) — Camera.java's wide intrinsics are commented out. If the device has a distinct wide camera, supply real wide intrinsics or `big_img` warp mis-registers. Validate by overlaying predicted laneLines on the road.
6. **`ModelExecutorExternal` still runs under `Runner==EXTERNAL_TINYGRAD`** — it does redundant image-prep and publishes to the legacy port 8228 (no port-5100 conflict, but wasted CPU and it sets the `ModelDReady` param the Python modeld relies on). Consider a dedicated "no Java executor" runner mode.
7. **Frame/state timestamp sync** — `roadCameraBuffer` and `roadCameraState` arrive on separate sockets; `read_frame` pairs the latest of each. Tighten if frame/metadata skew matters.
8. **Asset delivery** — `driving_supercombo.onnx` (fetch script) and the built `driving_tinygrad.pkl` must be present under `selfdrive/modeld/models/` on the device (the `.pkl` is backend-specific; build it on the target).
9. **Gradle build** — no Java was edited (`AndroidLauncher` already branches `EXTERNAL_TINYGRAD`), so no regression expected; still confirm `./gradlew :android:assembleDebug`.
10. **Dead code** — `fill_lane_line_meta`, `MIN_LAT_CONTROL_SPEED`, `v_ego` are now unused in the modeld path; harmless, clean up later.
