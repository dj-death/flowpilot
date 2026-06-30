# selfdrive/modeld/flowpilot_intrinsics.py
#
# Camera intrinsics transcribed from:
#   common/java/ai.flow.common/transformations/Camera.java
#
# Source values (default non-F2 device, FORCE_TELE_CAM_F3=false):
#   FocalX = FocalY = 1600.0f
#   CenterX = 952.62915f, CenterY = 517.53534f
#   frameSize = [1920, 1080]
#   digital_zoom_apply = 1600.0f / Model.MEDMODEL_FL (=910.0f) = 1600/910
#   OffsetX = CenterX - 960 = -7.37085
#   OffsetY = CenterY - 540 = -22.46466
#   cx = 960 + OffsetX * (1600/910) = 960 - 12.960 = 947.040
#   cy = 540 + OffsetY * (1600/910) = 540 - 39.498 = 500.502
#
# NOTE: Camera.java defines only one cam_intrinsics matrix. The commented-out
# WideIntrinsics (telephoto, 910f focal length) is not active. WIDE_INTRINSICS
# therefore reuses ROAD_INTRINSICS here. CONFIRM BOTH VALUES ON-DEVICE once
# the actual device camera is known (intrinsics are volatile and may be updated
# at runtime via Camera.updateIntrinsics()).
import numpy as np

ROAD_INTRINSICS = np.array([
    [1600.0,    0.0,  947.040],
    [   0.0, 1600.0,  500.502],
    [   0.0,    0.0,    1.0  ],
], dtype=np.float32)

# Same matrix reused; no separate wide-camera intrinsics are defined in Camera.java.
WIDE_INTRINSICS = np.array([
    [1600.0,    0.0,  947.040],
    [   0.0, 1600.0,  500.502],
    [   0.0,    0.0,    1.0  ],
], dtype=np.float32)
