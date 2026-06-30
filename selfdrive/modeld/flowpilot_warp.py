# selfdrive/modeld/flowpilot_warp.py
import numpy as np
from common.transformations.model import get_warp_matrix_intrinsics

def warp_matrices(rpy_calib, road_intrinsics, wide_intrinsics):
    rpy = np.asarray(rpy_calib, dtype=np.float32)
    tfm = get_warp_matrix_intrinsics(rpy, np.asarray(road_intrinsics, np.float32), False).astype(np.float32)
    big_tfm = get_warp_matrix_intrinsics(rpy, np.asarray(wide_intrinsics, np.float32), True).astype(np.float32)
    return tfm, big_tfm
