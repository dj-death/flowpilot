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
