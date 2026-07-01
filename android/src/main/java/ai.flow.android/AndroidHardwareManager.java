package ai.flow.android;

import ai.flow.hardware.HardwareManager;
import android.speech.tts.TextToSpeech;
import android.view.Window;
import android.view.WindowManager;

import java.util.Locale;

public class AndroidHardwareManager extends HardwareManager {
    public Window window;
    private TextToSpeech tts;
    private volatile boolean ttsReady = false;
    private volatile String desiredLang = "en";   // "en" | "fr" | "ar", from the VoiceLang setting

    public AndroidHardwareManager(Window window){
        this.window = window;
    }

    public void enableScreenWakeLock(boolean enable){
        if (enable)
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        else
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    @Override
    public void announce(String text){
        if (text == null || text.isEmpty())
            return;
        if (tts == null)
            initTts();
        if (ttsReady)
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "advisory");
    }

    private synchronized void initTts(){
        if (tts != null)
            return;
        try {
            tts = new TextToSpeech(window.getContext().getApplicationContext(), status -> {
                if (status == TextToSpeech.SUCCESS) {
                    ttsReady = true;
                    applyLanguage();
                }
            });
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // Set the spoken-advisory voice language from the "VoiceLang" setting ("en"/"fr"/"ar").
    @Override
    public void setVoiceLanguage(String lang){
        desiredLang = (lang == null || lang.isEmpty()) ? "en" : lang;
        if (ttsReady)
            applyLanguage();
    }

    private void applyLanguage(){
        if (tts == null)
            return;
        int r = tts.setLanguage(localeFor(desiredLang));
        if (r == TextToSpeech.LANG_MISSING_DATA || r == TextToSpeech.LANG_NOT_SUPPORTED)
            tts.setLanguage(Locale.US); // requested voice isn't installed on this device
    }

    private static Locale localeFor(String lang){
        switch (lang) {
            case "fr": return Locale.FRENCH;
            case "ar": return new Locale("ar");
            default:   return Locale.US;
        }
    }
}
