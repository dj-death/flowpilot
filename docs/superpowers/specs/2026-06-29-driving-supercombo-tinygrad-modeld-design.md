# Design: Add upstream `driving_supercombo` via an on-device tinygrad `modeld` co-process

- **Date:** 2026-06-29
- **Status:** Approved design (pre-plan)
- **Approach:** B — run upstream openpilot v0.11 `modeld.py` (tinygrad) as a Python co-process on-device, bridged to flowpilot over the existing cereal/ZMQ bus.
- **Scope decisions:** go straight to on-device (no desktop bring-up milestone); keep flowpilot's existing v0.9.4 planners (no control-stack/capnp changes).

## 1. Problem & goal

flowpilot currently runs an older openpilot driving model. The shipped artifact `selfdrive/assets/models/f3/supercombo.thneed` is a partially-migrated "LA" model executed by the Java `ModelExecutorF3` + native `jniconvert.cpp` (THNEED/OpenCL) path, with output parsed by native `selfdrive/modeld/modelparsed.cc` into `modelV2`/`cameraOdometry`.

Upstream openpilot `master` (v0.11.x) ships a new driving model, `driving_supercombo.onnx`. The goal is to run **that** model in flowpilot.

### Why this is not a drop-in

The new model's I/O is fundamentally different from flowpilot's current contract, **and** upstream no longer ships a `.thneed`/`.dlc` for it — it compiles the onnx to a **tinygrad pickle** and the on-device runtime is tinygrad itself. There is no maintained `onnx → thneed` exporter for this attention-based model, and the vendored `tinygrad_repo` (v0.7.0) is far too old to compile it.

We therefore adopt upstream's own runtime: ship the tinygrad pickle and run upstream `modeld.py` on-device as a co-process, reusing comma's warp/sample/recurrence/parse code unchanged, and bridge it to flowpilot over messaging.

## 2. Ground truth: `driving_supercombo.onnx` (read from the binary)

openpilot `master` @ commit `8c927d0f…`; model shipped in v0.11.0 (PR #36798). The onnx is a Git-LFS object on `gitlab.com/commaai/openpilot-lfs.git` (`sha256:659727c4…8009b`, 61 MB). A higher-capacity `big_driving_supercombo.onnx` (195 MB) shares the same I/O contract and is only used when a comma "USB GPU" accelerator is present — **out of scope**; we target `driving_supercombo.onnx`.

### Inputs (6)

| name | shape | dtype |
|---|---|---|
| `img` | `[1,12,128,256]` | uint8 |
| `big_img` | `[1,12,128,256]` | uint8 |
| `features_buffer` | `[1,24,512]` | float16 |
| `desire_pulse` | `[1,25,8]` | float16 |
| `traffic_convention` | `[1,2]` | float16 |
| `action_t` | `[1,2]` | float16 |

- The camera **warp is not in the onnx graph** — comma performs it in a separate tinygrad JIT (`make_warp`) that takes raw NV12 frames + 3×3 `tfm`/`big_tfm` matrices and emits the packed 6-channel YUV `img`/`big_img`. `tfm`/`big_tfm` are inputs to the warp JIT, **not** the onnx.
- `img`/`big_img` each hold 2 temporally-spaced frames (current + ~0.2 s old), 6 channels each → 12.
- `desire_pulse` is a rising-edge one-hot pulse (`DESIRE_LEN=8`), 100 frames @ 20 Hz max-pooled in groups of 4 → 25.
- `features_buffer` is 96 past features sampled `[::4]` → 24, refilled each frame from the model's own `hidden_state` output.
- `action_t = [lat_action_t, long_action_t]` actuation-time horizons (replaces v0.9.4's `lateral_control_params`/`prev_desired_curv`).
- **Removed vs v0.9.4:** `nav_features`, `nav_instructions`, `lateral_control_params`.

### Output (1): `outputs [1,2576]` float16

Sliced by an `output_slices` dict embedded in the onnx `metadata_props` (base64 pickle). Decoded:

| field | slice | len |
|---|---|---|
| `meta` | 0:55 | 55 |
| `desire_pred` | 55:87 | 32 |
| `pose` | 87:99 | 12 |
| `wide_from_device_euler` | 99:105 | 6 |
| `road_transform` | 105:117 | 12 |
| `lane_lines` | 117:645 | 528 |
| `lane_lines_prob` | 645:653 | 8 |
| `road_edges` | 653:917 | 264 |
| `lead` | 917:1061 | 144 |
| `lead_prob` | 1061:1064 | 3 |
| `hidden_state` | 1064:1576 | 512 |
| `plan` | 1576:2566 | 990 |
| `desire_state` | 2566:2574 | 8 |
| `pad` | -2: | 2 |

Plan is single-hypothesis (990 = 33×15 mu + std); lead is fixed 3-hypothesis (144). Recurrence: copy `outputs[1064:1576]` into the next `features_buffer`. There is **no committed `.pkl`** for the driving model — the slice metadata lives inside the onnx and is captured into the build artifact `driving_tinygrad.pkl` by `compile_modeld.py`.

## 3. Key codebase facts (flowpilot)

- **Messaging is interoperable Python↔Java.** Transport is ZMQ PUB/SUB over TCP loopback carrying raw capnp `Event` bytes, one port per service, ports derived identically by Java (`cereal/messaging/java/messaging/PortMap.java`, `STARTING_PORT=5100`) and Python (`cereal/services.py`) from the **same** `cereal/resources/services.yaml` ordering. `modelV2`→5100, `cameraOdometry`→5101. Python defaults to the ZMQ backend unless `MSGQ` is set.
- **flowpilot already runs upstream Python on-device.** `selfdrive/manager/flowinitd.py` + `process_config.py` launch `controlsd`, `plannerd`, `radard`, `calibrationd`, `modelparsed`, etc. via `subprocess`; on Android the JVM app boots from Termux (`LoadingActivity.java` → Termux `RUN_COMMAND`).
- **The model is already a co-process split.** `ModelExecutorF3` publishes `modelRaw`; `selfdrive/modeld/modelparsed.cc` subscribes and publishes `modelV2`+`cameraOdometry`. Our Python `modeld` takes over this publish surface.
- **Frames are available over ZMQ.** `CameraManager` publishes `roadCameraBuffer`/`wideRoadCameraBuffer` as `FrameBuffer` (`cereal/log.capnp:2176`) with the full YUV420 `image` blob + `stride/uOffset/vOffset/*PixelStride/frameWidth/Height` — self-describing, reconstructable cross-process (no VisionIpc). On Android the full pixels are on the wire; the desktop camera must run in **YUV mode** for cross-process frames (RGB mode uses a process-local pointer).
- **`EXTERNAL_TINYGRAD` precedent.** `common/java/ai.flow.common/utils.java` has the enum value (marked "DOESNT WORK"); `AndroidLauncher.java` has a commented-out Termux launch stub; `scripts/setup_tinygrad` builds a Termux Python env (pyopencl against Android `libOpenCL.so`, tinygrad from git) — rough/unmaintained.

## 4. Schema reconciliation (output side: essentially none)

flowpilot's `ModelDataV2` (`cereal/log.capnp:840`) and `cameraOdometry` (`:2032`) are **wire-compatible** with upstream v0.11 — matching Event-union ordinals (75/63), matching field numbers, identical `XYZTData` and `CameraOdometry` type IDs. Every field upstream `fill_model_msg.py` writes (position/velocity/acceleration/orientation/orientationRate, laneLines/probs/stds, roadEdges, leadsV3, meta{engagedProb,desireState,desirePrediction,disengagePredictions,hardBrakePredicted}, confidence, pose fields) has a compatible slot. capnp forward-compat absorbs the rest.

Feeding flowpilot's consumers (Python `lateral_planner`/`longitudinal_planner`/`radard`/`controlsd`, Java `OnRoadScreen`/`MsgModelDataV2`) upstream-shaped messages breaks **nothing**; two harmless graceful degradations:
- `temporalPose` not written → `drive_helpers.get_speed_error` returns 0 (model speed-error compensation disabled). Optional to restore.
- `lateralPlannerSolution` not written → already dead code in `lateral_planner.py`.

**No capnp edits, no `Definitions.java` regen, no consumer edits required.**

## 5. modeld input reconciliation (the only real input work)

Upstream `modeld.py` SubMaster: `deviceState, carState, roadCameraState, liveCalibration, driverMonitoringState, carControl, liveDelay`.

| input | flowpilot status | action |
|---|---|---|
| `liveCalibration.rpyCalib` | ✅ compatible | use directly |
| `carState.vEgo` | ✅ compatible | use directly |
| `carControl.latActive` | ✅ compatible (ordinal+field match) | use directly |
| `carParams.longitudinalActuatorDelay` | ✅ flowpilot's `@58` (`…UpperBound`) is the **same field number** | use directly (add code comment) |
| `deviceState`, `roadCameraState` | ✅ ordinals match; `DEVICE_CAMERAS` lookup may need a flowpilot entry | add device/sensor entry if needed |
| `driverMonitoringState.isRHD` | ❌ incompatible struct (union 71 vs 151, field 4 vs 7) | **stub `is_rhd = False`**, drop from SubMaster |
| `liveDelay.lateralDelay` | ❌ service absent in flowpilot | **stub `lat_delay = 0.2`**, drop from SubMaster |

## 6. Architecture & data flow

```
 camera (Java)                                 Python modeld (tinygrad)
 ──────────────                                ────────────────────────
 road/wideRoadCameraBuffer (FrameBuffer YUV) ───►  sub: frames
 liveCalibration ────────────────────────────►   sub: rpyCalib → tfm/big_tfm
 carState / carControl / carParams ──────────►   sub: vEgo, latActive, longActuatorDelay
                                                  │
                                                  ├─ warp JIT (make_warp): NV12 + tfm → img/big_img (uint8)
                                                  ├─ temporal sample: desire_pulse(25,8), features_buffer(24,512)
                                                  ├─ run_policy (tinygrad pickle): driving_supercombo
                                                  ├─ parse_model_outputs + fill_model_msg
                                                  └─ pub: modelV2 (5100), cameraOdometry (5101)
                                                          │
 OnRoadScreen (Java UI) ◄────────────────────────────────┤
 plannerd / controlsd / radard / calibrationd (Python) ◄──┘
```

The Java `ModelExecutorF3` and native `modelparsed.cc` are **disabled** (single publisher on 5100).

## 7. Components / work breakdown

1. **Vendor upstream modeld stack** into flowpilot, wired to **this repo's `cereal`**: `modeld.py`, `parse_model_outputs.py`, `fill_model_msg.py`, `constants.py`, `helpers.py`, `compile_modeld.py`, and the needed `common/transformations/*` (`get_warp_matrix`, model intrinsics). Keep `MSGQ` unset.
2. **Upgrade tinygrad** from the vendored v0.7.0 to the upstream-pinned `f9c8c697` (or pip), sufficient to load/run the pickle and (if compiling on-device) run `compile_modeld.py`.
3. **Model asset pipeline**: fetch `driving_supercombo.onnx` (GitLab LFS); produce `driving_tinygrad.pkl` for the target backend. Pickle is backend-specific → compile on-device (first run) or prebuild for the Adreno/OpenCL backend. Ship via flowpilot's asset mechanism.
4. **`modeld.py` adaptations**:
   - Replace `VisionIpcClient` with a flowpilot frame subscriber (`roadCameraBuffer`/`wideRoadCameraBuffer`); reconstruct images from `FrameBuffer.image` + geometry; pull `frameId`/timestamps.
   - Drive `tfm`/`big_tfm` from `liveCalibration.rpyCalib` + flowpilot camera intrinsics (`get_warp_matrix`); ensure intrinsics/`DEVICE_CAMERAS` parity with flowpilot's camera config.
   - Stub `is_rhd = False`, `lat_delay = 0.2`; drop `driverMonitoringState`/`liveDelay` from SubMaster.
   - Publish `modelV2`+`cameraOdometry`; drop unused `drivingModelData` pub. (Optional: write `temporalPose`.)
5. **Process management & runner gating**: add `modeld` as a `ManagerProcess` in `process_config.py`; introduce a runner mode (reuse/repair `EXTERNAL_TINYGRAD`) that prevents `ModelExecutorF3` + `modelparsed` from starting and launches the Python modeld under Termux.
6. **On-device tinygrad runtime**: repair `scripts/setup_tinygrad` (Termux Python, pyopencl vs Android `libOpenCL.so`, tinygrad install); select the Adreno GPU (OpenCL) backend.

## 8. Milestones (on-device, de-risking order)

- **M1 — On-device tinygrad runtime + asset.** Repair the Termux Python/tinygrad/pyopencl env; fetch onnx; produce the pickle; run inference standalone on the phone GPU on a static input. **Gate: measure inference rate** (the dominant unknown). This is sequenced first because it is the highest-risk, most independent piece.
- **M2 — Vendor + adapt `modeld.py`.** Frame bridging, input stubs, warp/intrinsics parity, publish `modelV2`/`cameraOdometry`. (Code is device-independent; validated on-device.)
- **M3 — On-device integration.** Manager launches modeld; Java executor + `modelparsed` gated off; end-to-end run on the phone; validate UI + planners consume outputs; **measure frame throughput/latency**.
- **M4 — Perf/quality hardening.** Optimize toward a usable frame rate; tune conflate/throughput; verify calibration registration.

## 9. Risks

1. **On-device tinygrad runtime (highest).** `EXTERNAL_TINYGRAD`/`setup_tinygrad` are rough/unmaintained; getting the attention model to a usable rate (target ~20 Hz, road model) on the phone GPU is the make-or-break engineering risk. Surfaced first by M1.
2. **Frame throughput over ZMQ TCP.** ~62 MB/s at 1080p/20 Hz with cross-process copies (no shared-memory VisionIpc). May need conflate, resolution, or transport tuning. Measured in M3.
3. **Preprocessing/calibration parity.** The Python warp must match flowpilot's intrinsics/calibration exactly or outputs mis-register against the UI/planners. Validated in M2/M3.

## 10. Explicit non-goals

- Native `.thneed`/`.dlc` for the new model (no maintained exporter — that was Approach C, rejected).
- A desktop ONNX-Runtime deliverable (Approach A, rejected).
- Control-stack changes / v0.11 end-to-end curvature (`action.desiredCurvature`) — keep v0.9.4 planners.
- `big_driving_supercombo.onnx` / USB-GPU accelerator support.
- capnp schema or `Definitions.java` changes (none needed).

## 11. References

- Upstream: `commaai/openpilot@master:openpilot/selfdrive/modeld/{modeld.py,compile_modeld.py,parse_model_outputs.py,fill_model_msg.py,constants.py,helpers.py,get_model_metadata.py,models/README.md}`; `common/transformations/{model,camera,orientation}.py`; onnx `sha256:659727c4…8009b`.
- flowpilot: `selfdrive/modeld/java/ai.flow.modeld/{ModelExecutorF3,CommonModelF3}.java`; `selfdrive/modeld/modelparsed.cc`; `android/src/main/cpp/jniconvert.cpp`; `cereal/messaging/**`, `cereal/services.py`, `cereal/resources/services.yaml`, `cereal/log.capnp`, `cereal/car.capnp`; `selfdrive/manager/{flowinitd,process_config}.py`; `android/src/main/java/ai.flow.android/{AndroidLauncher,LoadingActivity,sensor/CameraManager}.java`; `common/java/ai.flow.common/{utils,Path,transformations/Camera}.java`; `scripts/setup_tinygrad`; `tinygrad_repo/` (v0.7.0).
