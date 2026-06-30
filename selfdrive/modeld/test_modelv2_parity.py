# selfdrive/modeld/test_modelv2_parity.py
import numpy as np, onnxruntime as ort, pickle
from selfdrive.modeld.parse_model_outputs import Parser
from selfdrive.modeld.constants import ModelConstants

ONNX = "selfdrive/assets/models/driving/driving_supercombo.onnx"
PKL  = "selfdrive/assets/models/driving/driving_tinygrad.pkl"

def test_parse_shapes():
    md = pickle.load(open(PKL, "rb"))["metadata"]
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    feeds = {}
    for i in sess.get_inputs():
        dt = np.float16 if i.type == 'tensor(float16)' else np.uint8
        feeds[i.name] = np.zeros([d for d in i.shape], dtype=dt)
    out = sess.run(None, feeds)[0][0]                      # (2576,) fp16
    sliced = {k: out[np.newaxis, v] for k, v in md["output_slices"].items()}
    parsed = Parser().parse_outputs(sliced)
    assert parsed["plan"].shape[1:] == (ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH)
    assert "lane_lines" in parsed and "lead" in parsed
    print({k: np.asarray(v).shape for k, v in parsed.items()})
