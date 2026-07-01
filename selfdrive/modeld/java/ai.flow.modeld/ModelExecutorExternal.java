package ai.flow.modeld;

import ai.flow.common.ParamsInterface;
import ai.flow.definitions.Definitions;

// EXTERNAL_TINYGRAD runner: the driving model runs entirely out-of-process in the Python tinygrad
// modeld (selfdrive/modeld/modeld_flowpilot.py), which subscribes to roadCameraBuffer directly,
// runs driving_supercombo on the GPU and publishes modelV2. The Java side only needs to publish
// camera frames (CameraManager) and let the UI consume modelV2, so this executor is a no-op that
// just marks the model ready. (Previously it did redundant Java-side image prep and shipped tensors
// to a legacy port 8228 with the actual run commented out -- dead weight, and a crash risk.)
public class ModelExecutorExternal extends ModelExecutor {

    public boolean stopped = true;
    public boolean initialized = false;
    public final ParamsInterface params = ParamsInterface.getInstance();

    public ModelExecutorExternal() {
        instance = this;
    }

    @Override
    public void ExecuteModel(Definitions.FrameData.Reader wideData, Definitions.FrameBuffer.Reader wideBuf,
                             long processStartTimestamp) {
        frameWideData = frameData = wideData;
        msgFrameWideBuffer = msgFrameBuffer = wideBuf;
        // inference happens out-of-process in the Python tinygrad modeld
    }

    @Override
    public void init() {
        if (initialized) return;
        initialized = true;
        params.putBool("ModelDReady", true);
    }

    @Override
    public boolean isRunning() { return !stopped; }

    @Override
    public boolean isInitialized() { return initialized; }

    @Override
    public void dispose() { stopped = true; }

    @Override
    public void stop() { stopped = true; }

    @Override
    public void start() { stopped = false; }
}
