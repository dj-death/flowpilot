package ai.flow.android.sensor;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Context;
import android.content.pm.PackageManager;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;

import ai.flow.common.SpeedData;
import ai.flow.sensor.SensorInterface;

/**
 * GPS ground-speed source for the onroad speed estimate.
 *
 * With no OBD / panda there is no vehicle CAN speed, so OnRoadScreen fuses the model's
 * predicted ego-velocity (fast, smooth) with an absolute GPS speed (this class) to show a
 * vehicle speed anyway. Read-only: the value never feeds the control path.
 *
 * Uses the platform LocationManager (no Google Play Services dependency). Speed comes from
 * Location.getSpeed() on GPS_PROVIDER (Doppler-derived). Registration retries until the
 * ACCESS_FINE_LOCATION runtime permission is granted, mirroring ELM327Manager's retry loop.
 */
public class GpsManager extends SensorInterface {

    private static final long MIN_TIME_MS = 500;   // requested location cadence
    private static final float MIN_DIST_M = 0f;
    private static final long RETRY_MS = 3000;     // permission/registration retry backoff

    private final Context context;
    private volatile boolean running = false;
    private HandlerThread thread;
    private Handler handler;
    private LocationManager locationManager;
    private LocationListener listener;

    public GpsManager(Context context) {
        this.context = context.getApplicationContext();
    }

    @Override
    public boolean isRunning() {
        return running;
    }

    @Override
    public void start() {
        if (running)
            return;
        running = true;
        SpeedData.gpsEnabled = true;
        thread = new HandlerThread("GpsManager");
        thread.start();
        handler = new Handler(thread.getLooper());
        handler.post(this::register);
    }

    @SuppressLint("MissingPermission")
    private void register() {
        if (!running)
            return;
        if (context.checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) {
            SpeedData.statusMessage = "GPS: waiting for location permission";
            handler.postDelayed(this::register, RETRY_MS); // retry until granted
            return;
        }
        try {
            locationManager = (LocationManager) context.getSystemService(Context.LOCATION_SERVICE);
            if (locationManager == null) {
                SpeedData.statusMessage = "GPS: no location service";
                return;
            }
            listener = new LocationListener() {
                @Override
                public void onLocationChanged(Location location) {
                    if (location.hasSpeed()) {
                        SpeedData.gpsSpeedMps = location.getSpeed();
                        SpeedData.gpsLastMs = System.currentTimeMillis();
                        SpeedData.gpsHasFix = true;
                        SpeedData.statusMessage = "GPS: fix";
                    }
                }

                @Override public void onStatusChanged(String provider, int status, Bundle extras) {}
                @Override public void onProviderEnabled(String provider) {}
                @Override public void onProviderDisabled(String provider) {
                    SpeedData.gpsHasFix = false;
                    SpeedData.statusMessage = "GPS: provider disabled";
                }
            };
            locationManager.requestLocationUpdates(LocationManager.GPS_PROVIDER,
                    MIN_TIME_MS, MIN_DIST_M, listener, thread.getLooper());
            SpeedData.statusMessage = "GPS: waiting for fix";
        } catch (SecurityException e) {
            SpeedData.statusMessage = "GPS: permission denied";
            handler.postDelayed(this::register, RETRY_MS);
        } catch (Exception e) {
            SpeedData.statusMessage = "GPS: " + e.getMessage();
        }
    }

    @Override
    public void stop() {
        running = false;
        SpeedData.gpsHasFix = false;
        try {
            if (locationManager != null && listener != null)
                locationManager.removeUpdates(listener);
        } catch (Exception ignored) {}
        if (thread != null)
            thread.quitSafely();
    }

    @Override
    public void dispose() {
        stop();
    }
}
