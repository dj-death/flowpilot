# selfdrive/modeld/bench_driving_supercombo.py
import pickle, time, numpy as np
from tinygrad.tensor import Tensor
from selfdrive.modeld.compile_modeld import make_input_queues, WARP_INPUTS, POLICY_INPUTS
from system.camerad.cameras.nv12_info import get_nv12_info

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
