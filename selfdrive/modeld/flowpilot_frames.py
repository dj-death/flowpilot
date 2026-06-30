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
