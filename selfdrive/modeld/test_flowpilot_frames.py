# selfdrive/modeld/test_flowpilot_frames.py
import numpy as np
from system.camerad.cameras.nv12_info import get_nv12_info
from selfdrive.modeld.flowpilot_frames import to_model_nv12

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
