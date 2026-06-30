# selfdrive/modeld/drive_modeld_synthetic.py
# Synthetic camera/state publisher to drive modeld_flowpilot end-to-end (no Java app),
# and verify it publishes modelV2. Run AFTER modeld_flowpilot is up on the bus.
import time
import numpy as np
import cereal.messaging as messaging

W, H = 1920, 1080
YS = W * H
UVS = W * H // 2
_img = np.zeros(YS + UVS, dtype=np.uint8)
_img[:YS] = (np.arange(YS) % 256).astype(np.uint8)   # faint gradient, not all-zero
IMG = _img.tobytes()

def fill_buf(fb):
    fb.image = IMG
    fb.frameWidth = W
    fb.frameHeight = H
    fb.stride = W
    fb.uvPixelStride = 2
    fb.uOffset = YS
    fb.vOffset = YS + 1

def main():
    pm = messaging.PubMaster(["roadCameraBuffer", "wideRoadCameraBuffer",
                              "roadCameraState", "wideRoadCameraState",
                              "liveCalibration", "carState", "carControl"])
    sm = messaging.SubMaster(["modelV2", "cameraOdometry"])
    time.sleep(1.2)
    got = False
    for i in range(200):
        t = int(time.monotonic() * 1e9)
        rs = messaging.new_message("roadCameraState")
        rs.roadCameraState.frameId = i; rs.roadCameraState.timestampSof = t; rs.roadCameraState.timestampEof = t
        pm.send("roadCameraState", rs)
        ws = messaging.new_message("wideRoadCameraState")
        ws.wideRoadCameraState.frameId = i; ws.wideRoadCameraState.timestampSof = t; ws.wideRoadCameraState.timestampEof = t
        pm.send("wideRoadCameraState", ws)
        lc = messaging.new_message("liveCalibration")
        lc.liveCalibration.rpyCalib = [0.0, 0.0, 0.0]
        pm.send("liveCalibration", lc)
        cs = messaging.new_message("carState"); cs.carState.vEgo = 10.0; pm.send("carState", cs)
        cc = messaging.new_message("carControl"); cc.carControl.latActive = True; pm.send("carControl", cc)
        rb = messaging.new_message("roadCameraBuffer"); fill_buf(rb.roadCameraBuffer); pm.send("roadCameraBuffer", rb)
        wb = messaging.new_message("wideRoadCameraBuffer"); fill_buf(wb.wideRoadCameraBuffer); pm.send("wideRoadCameraBuffer", wb)
        time.sleep(0.05)
        sm.update(0)
        if sm.updated["modelV2"]:
            m = sm["modelV2"]
            px = list(m.position.x)[:3]
            print(f"GOT modelV2: frameId={m.frameId} nLaneLines={len(m.laneLines)} position.x[:3]={px}")
            got = True
            break
    print("MODELV2_RECEIVED" if got else "NO_MODELV2_after_200_frames")

if __name__ == "__main__":
    main()
