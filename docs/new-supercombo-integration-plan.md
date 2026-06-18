# Integrating a newer openpilot supercombo into flowpilot

## TL;DR

flowpilot currently runs an **openpilot ~0.8.x-era supercombo** model as a precompiled
**thneed** (OpenCL) blob, with a Java model harness whose input/output layout is **hardcoded**
for that specific model. Upgrading to a newer upstream supercombo is **not a drop-in**: it
requires obtaining and recompiling the model for the phone GPU, rewriting the output parser
to the new tensor layout, adjusting the model inputs, and reconciling the downstream
planner/controls contract. None of these can be validated without a physical device + the
model binary, so this document is the execution plan rather than a code change.

## Current state (what's coupled to the model)

| Area | File | Coupling |
|---|---|---|
| Runtime format | `ModelExecutorF3`, `*ModelRunner`, `selfdrive/assets/models/f3/supercombo.thneed` | Default runner is **THNEED** (compiled OpenCL). `.dlc` = SNPE. ONNX/TNN/tinygrad runners are present but marked "DOESNT WORK". |
| Output size | `CommonModelF3.OUTPUT_SIZE=5992`, `NET_OUTPUT_SIZE=6504`, `FEATURE_LEN=512` | Fixed scalars for the old model. |
| Output parsing | `Parser.java` (~410 lines), `ParsedOutputs` | Slices the flat output array at **fixed offsets** for the old layout (plan, lane lines, road edges, leads MDN, meta, pose). |
| Inputs | `ModelExecutorF3.init()` | Feeds `input_imgs`, `big_input_imgs`, `desire`, `traffic_convention`, `features_buffer`, **`nav_features`**, **`nav_instructions`**. |
| Image prep | `ImagePrepareGPU/CPU`, `Preprocess` | 512×256, 2-frame YUV420, warp from camera intrinsics. |

## What changed upstream (the gap)

Current upstream ships `driving_supercombo.onnx` / `big_driving_supercombo.onnx` (ONNX), with:

1. **Output layout is different** and is described by a **metadata file** (`slice_outputs` /
   a metadata pkl), not the fixed offsets flowpilot hardcodes.
2. **Inputs changed**: the **nav inputs were removed**; the policy side uses a
   previous-desired-curvature input and an evolved feature buffer.
3. **Output semantics changed**: the model emits driving-policy outputs (e.g. desired
   curvature) that the planners consume differently than the 0.8.x path flowpilot uses.

## Required work (ordered)

1. **Obtain the model + metadata.** Pull the target `driving_supercombo.onnx` and its output
   metadata from the matching upstream commit. Decide F2 vs F3 (medium vs big) mapping.
2. **Get it running on the phone GPU.** flowpilot's ONNX path is non-functional, so the model
   must be compiled to **thneed** for the Adreno GPU using openpilot's compiler
   (tinygrad `compile2.py`) on a Snapdragon device, producing a new `supercombo.thneed`.
   (Alternatively, get the ONNX/SNPE runner working — larger effort.)
3. **Update I/O constants** in `CommonModelF3` (`OUTPUT_SIZE`, `NET_OUTPUT_SIZE`, `FEATURE_LEN`,
   `T_IDXS`/`X_IDXS` if changed).
4. **Rewrite `Parser.java`** to the new output layout. Strongly prefer a **metadata-driven
   slicing** approach (mirror upstream `slice_outputs`) over fixed offsets so future model
   bumps are config, not code.
5. **Update inputs** in `ModelExecutorF3.init()`: drop `nav_features`/`nav_instructions` if the
   target model removed them; add prev-desired-curvature; match `features_buffer` shape.
6. **Reconcile downstream**: update planner/controls consumers for any changed output semantics
   (e.g. model-provided curvature). This is the same divergence handled case-by-case in the
   controls port analysis (`docs/controls-upstream-port-analysis.md`).
7. **Validate on-device**: visualize lane lines/leads/path on a real drive; compare against the
   old model before trusting it for control.

## Why this isn't being done as a code change here

- The model binary is a large LFS asset and the thneed compilation requires a Snapdragon GPU
  toolchain — neither is available in this environment.
- The output parser is the most safety-critical glue in the app; writing it blind against a
  layout I cannot run or diff would be unsafe.
- No build (no Android SDK) and no inference means zero verification.

## Recommended next step

Pick a concrete target model version (which upstream commit / F2 vs F3), provide the compiled
`supercombo.thneed` (or a device to compile on), and then the parser/IO rewrite (steps 3–6)
can be done and verified against it. The metadata-driven parser refactor (step 4) is the
highest-leverage piece and could be prototyped first.
