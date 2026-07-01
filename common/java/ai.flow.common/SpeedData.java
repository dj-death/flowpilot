package ai.flow.common;

/**
 * Shared, read-only snapshot of the vehicle-speed estimate used when there is no OBD /
 * vehicle-CAN speed source.
 *
 * The android module's GpsManager writes the GPS fields; the UI module's OnRoadScreen reads
 * them and fuses them with the model's predicted ego-velocity (modelV2.velocity). This holder
 * lives in the common module so both sides can reference it without a circular module
 * dependency, mirroring {@link OBDData}.
 *
 * Display-only: nothing here ever feeds the openpilot control path.
 */
public class SpeedData {
    public static volatile boolean gpsEnabled = false;      // GPS speed source was requested
    public static volatile boolean gpsHasFix = false;       // a location carrying speed has arrived
    public static volatile float gpsSpeedMps = Float.NaN;   // last GPS ground speed (m/s)
    public static volatile long gpsLastMs = 0;              // when gpsSpeedMps was last set
    public static volatile String statusMessage = "GPS: off";
}
