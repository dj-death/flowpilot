# selfdrive/modeld/check_modelv2_live.py
import time
from cereal import messaging
sm = messaging.SubMaster(["modelV2", "cameraOdometry"])
t0 = time.monotonic(); n = 0
while time.monotonic() - t0 < 10:
    sm.update(100)
    if sm.updated["modelV2"]:
        n += 1
        md = sm["modelV2"]
        assert len(md.position.x) > 0, "empty position"
print(f"modelV2 msgs in 10s: {n}  ({n/10:.1f} Hz)")
assert n > 0, "no modelV2 received"
