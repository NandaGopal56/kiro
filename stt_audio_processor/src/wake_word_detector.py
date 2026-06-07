"""
wake_word_detector.py
----------------------
Simple substring-based wake word gate.

The detector has two jobs:
  1. check_for_wake_word(text) — scan a transcript for any wake word
  2. Track whether the session is still "active" (within idle timeout)

The idle timeout is now managed by VoiceProcessor directly via
SESSION_IDLE_TIMEOUT, so this class is intentionally thin — it just
holds the word list and active flag.

To upgrade to a proper model (Porcupine, openWakeWord), replace
check_for_wake_word() only — nothing else changes.
"""

from stt_audio_processor.utils.config import WAKE_WORD_CONFIG


class WakeWordDetector:

    def __init__(self):
        self.wake_words = [w.lower() for w in WAKE_WORD_CONFIG.wake_words]
        self.is_active = False

    def check_for_wake_word(self, text: str) -> bool:
        """
        Return True if any wake word appears in text (case-insensitive).
        Also sets is_active=True so callers can check state.
        """
        lowered = text.lower()
        for word in self.wake_words:
            if word in lowered:
                print(f"[WakeWord] Detected '{word}' in: \"{text}\"")
                self.is_active = True
                return True
        return False

    def deactivate(self):
        self.is_active = False
        print("[WakeWord] Session ended - back to sleep.")