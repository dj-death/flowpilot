#!/usr/bin/env python3
import os
os.environ['GMMU'] = '0' # for usbgpu fast loading, noop for qcom
from tinygrad.tensor import Tensor
import time
import pickle
import numpy as np
import cereal.messaging as messaging
from cereal import log
from cereal import car
from cereal.messaging import PubMaster, SubMaster
from system.swaglog import cloudlog
from common.params import Params
from common.filter_simple import FirstOrderFilter
from common.realtime import config_realtime_process, DT_MDL
from system.camerad.cameras.nv12_info import get_nv12_info
from selfdrive.controls.lib.desire_helper import DesireHelper
from selfdrive.modeld.parse_model_outputs import Parser
from selfdrive.modeld.compile_modeld import make_input_queues, WARP_INPUTS, POLICY_INPUTS
from selfdrive.modeld.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState
from common.file_chunker import read_file_chunked, get_manifest_path
from selfdrive.modeld.constants import ModelConstants, Plan
from selfdrive.modeld.helpers import usbgpu_present, modeld_pkl_path, get_tg_input_devices

PROCESS_NAME = "openpilot.selfdrive.modeld.modeld"
SEND_RAW_PRED = os.getenv('SEND_RAW_PRED')

LAT_SMOOTH_SECONDS = 0.0
LONG_SMOOTH_SECONDS = 0.3
MIN_LAT_CONTROL_SPEED = 0.3


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


class ModelState:
  prev_desire: np.ndarray  # for tracking the rising edge of the pulse

  def __init__(self, cam_w: int, cam_h: int, usbgpu: bool):
    input_devices = get_tg_input_devices(PROCESS_NAME, usbgpu)
    self.WARP_DEV, self.QUEUE_DEV = input_devices['WARP_DEV'], input_devices['QUEUE_DEV']
    jits = pickle.loads(read_file_chunked(modeld_pkl_path(usbgpu)))
    metadata = jits['metadata']
    self.input_shapes = metadata['input_shapes']
    self.vision_input_names = [k for k in self.input_shapes if 'img' in k]
    self.output_slices = metadata['output_slices']

    self.prev_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)

    self.frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
    self.input_queues, self.npy = make_input_queues(self.input_shapes, self.frame_skip, device=self.QUEUE_DEV)
    self.full_frames: dict[str, Tensor] = {}
    self._blob_cache: dict[int, Tensor] = {}
    self.parser = Parser()
    self.frame_buf_params = {k: get_nv12_info(cam_w, cam_h) for k in ('img', 'big_img')}
    self.run_policy = jits['run_policy']
    self.warp = jits[(cam_w,cam_h)]

  def slice_outputs(self, model_outputs: np.ndarray, output_slices: dict[str, slice]) -> dict[str, np.ndarray]:
    parsed_model_outputs = {k: model_outputs[np.newaxis, v] for k,v in output_slices.items()}
    return parsed_model_outputs

  def run(self, bufs: dict[str, np.ndarray], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    for key in bufs.keys():
      # flowpilot frames are fresh bytes each iteration (no VisionBuf ringbuffer),
      # so build a fresh device tensor per frame instead of caching raw pointers
      # (the upstream from_blob/_blob_cache would leak and dangle on freed bytes).
      arr = np.frombuffer(bufs[key].data, dtype=np.uint8).copy()
      self.full_frames[key] = Tensor(arr, device=self.WARP_DEV)

    # Model decides when action is completed, so desire input is just a pulse triggered on rising edge
    inputs['desire_pulse'][0] = 0
    self.npy['desire'][:] = np.where(inputs['desire_pulse'] - self.prev_desire > .99, inputs['desire_pulse'], 0)
    self.prev_desire[:] = inputs['desire_pulse']
    self.npy['traffic_convention'][:] = inputs['traffic_convention']
    self.npy['action_t'][:] = inputs['action_t']
    self.npy['tfm'][:,:] = transforms['img'][:,:]
    self.npy['big_tfm'][:,:] = transforms['big_img'][:,:]

    warped = self.warp(**{k: self.input_queues[k] for k in WARP_INPUTS}, frame=self.full_frames['img'], big_frame=self.full_frames['big_img'])

    outs, = self.run_policy(
      **{k: self.input_queues[k] for k in POLICY_INPUTS if k in self.input_queues}, warped=warped
    )
    model_output = outs.numpy()[0]
    outputs_dict = self.parser.parse_outputs(self.slice_outputs(model_output, self.output_slices))
    self.npy['prev_feat'][:] = model_output[self.output_slices['hidden_state']]

    if SEND_RAW_PRED:
      outputs_dict['raw_pred'] = model_output.copy()
    return outputs_dict


def main(demo=False):
  cloudlog.warning("modeld init")

  _present = usbgpu_present()
  _compiled = os.path.isfile(get_manifest_path(modeld_pkl_path(usbgpu=True)))
  USBGPU = _present and _compiled
  params = Params()

  config_realtime_process(7, 54)

  # flowpilot frame subscribers (no VisionIPC)
  frame_sub = messaging.SubMaster(["roadCameraState", "wideRoadCameraState"])
  road_buf_sock = messaging.sub_sock("roadCameraBuffer", conflate=True)
  wide_buf_sock = messaging.sub_sock("wideRoadCameraBuffer", conflate=True)
  main_wide_camera = False
  use_extra_client = True
  CAM_W, CAM_H = 1920, 1080
  st = time.monotonic()
  model = ModelState(CAM_W, CAM_H, USBGPU)
  cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s, modeld starting")

  # messaging
  pm = PubMaster(["modelV2", "cameraOdometry"])
  sm = SubMaster(["carState", "liveCalibration", "carControl"])

  publish_state = PublishState()
  params = Params()

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / ModelConstants.MODEL_RUN_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()

  CP = car.CarParams.from_bytes(params.get("CarParams", block=True))
  cloudlog.info("modeld got CarParams: %s", CP.carName)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  # TODO Move smooth seconds to action function
  long_delay = CP.longitudinalActuatorDelayUpperBound + LONG_SMOOTH_SECONDS

  DH = DesireHelper()

  from selfdrive.modeld.flowpilot_frames import read_frame
  # recv_one_or_none is non-blocking + conflate, so buffers arrive on different loop iterations;
  # latch each until consumed. flowpilot devices with a single (road-only) camera never publish
  # the wide buffer, so process on the road frame and reuse it for the wide input.
  last_rb, last_wb = None, None
  while True:
    rb = messaging.recv_one_or_none(road_buf_sock)
    wb = messaging.recv_one_or_none(wide_buf_sock)
    if rb is not None: last_rb = rb
    if wb is not None: last_wb = wb
    frame_sub.update(0)
    if last_rb is None:
      continue
    rb, last_rb = last_rb, None
    wb, last_wb = last_wb, None
    buf_main = read_frame(rb.roadCameraBuffer, frame_sub["roadCameraState"])
    if wb is not None:
      buf_extra = read_frame(wb.wideRoadCameraBuffer, frame_sub["wideRoadCameraState"])
    else:
      buf_extra = read_frame(rb.roadCameraBuffer, frame_sub["roadCameraState"])  # single-camera: reuse road
    meta_main, meta_extra = buf_main, buf_extra   # FlowpilotBuf carries frame_id/timestamps

    sm.update(0)
    desire = DH.desire
    is_rhd = False                                   # flowpilot has no compatible driverMonitoringState
    frame_id = meta_main.frame_id
    v_ego = max(sm["carState"].vEgo, 0.)
    lat_delay = 0.2 + LAT_SMOOTH_SECONDS             # flowpilot has no liveDelay service
    if sm.updated["liveCalibration"]:
      from selfdrive.modeld.flowpilot_warp import warp_matrices
      from selfdrive.modeld.flowpilot_intrinsics import ROAD_INTRINSICS, WIDE_INTRINSICS
      rpy = np.array(sm["liveCalibration"].rpyCalib, dtype=np.float32)
      model_transform_main, model_transform_extra = warp_matrices(rpy, ROAD_INTRINSICS, WIDE_INTRINSICS)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(ModelConstants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < ModelConstants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}
    frame_delay = DT_MDL # compensate for time passed since the frame was captured: current_time - timestamp_eof is 50ms on average
    action_delay = DT_MDL / 2 # middle of the interval between model output (current state) and next frame (expected state)
    lat_action_t = lat_delay + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay
    inputs: dict[str, np.ndarray] = {
      'desire_pulse': vec_desire,
      'traffic_convention': traffic_convention,
      'action_t': np.array([lat_action_t, long_action_t], dtype=np.float32),
    }

    mt1 = time.perf_counter()
    model_output = model.run(bufs, transforms, inputs)
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      posenet_send = messaging.new_message('cameraOdometry')

      fill_model_msg(modelv2_send, model_output,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen)

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.LateralPlan.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.LateralPlan.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob)

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('cameraOdometry', posenet_send)
    last_vipc_frame_id = meta_main.frame_id


if __name__ == "__main__":
  try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning("got SIGINT")
