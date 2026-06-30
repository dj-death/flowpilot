# selfdrive/modeld/policy_bench.py  -- minimal policy-JIT timer for A/B comparison
# Feeds `warped` as a fresh random tensor (matching how compile_jit captured the JIT),
# so it avoids the warp->policy buffer mismatch in bench_driving_supercombo.py.
import sys, pickle, time, numpy as np
from tinygrad.tensor import Tensor
from selfdrive.modeld.compile_modeld import make_input_queues, POLICY_INPUTS, make_random_images

def main(pkl, n=60):
    d = pickle.load(open(pkl, "rb"))
    md, run_policy = d["metadata"], d["run_policy"]
    iq, npy = make_input_queues(md["input_shapes"], 4, device=None)
    img_hw = md["input_shapes"]["img"][2:]
    warped = make_random_images(keys=["warped"], shape=(2, 6, *img_hw))["warped"].realize()
    ts = []
    for i in range(n):
        for v in npy.values():
            v[:] = np.random.randn(*v.shape).astype(v.dtype)
        t0 = time.perf_counter()
        out, = run_policy(**{k: iq[k] for k in POLICY_INPUTS}, warped=warped)
        _ = out.numpy()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts[10:])
    name = pkl.split("/")[-1]
    print(f"POLICY {name}: mean {ts.mean()*1e3:.1f}ms  p50 {np.percentile(ts,50)*1e3:.1f}ms  -> {1/ts.mean():.1f} Hz")

if __name__ == "__main__":
    main(sys.argv[1])
