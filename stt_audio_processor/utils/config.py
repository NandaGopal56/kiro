from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    sample_rate: int   = 16000
    encoding: str      = "audio/wav"
    language_code: str = "en-IN"

    # Silence detection — RMS below this = silence.
    # Run recalibrate() to measure your room's ambient RMS and set this
    # slightly above it. Typical quiet room: 50-150. Noisy room: 200-400.
    silence_rms_threshold: int = 200

    # How long RMS stays below threshold before EOU fires (seconds)
    eou_silence_timeout: float  = 2.0
    # Burst length for wake word detection (seconds)
    wake_burst_seconds: float   = 2.0
    # Total silence before going back to sleep (seconds)
    session_idle_timeout: float = 30.0


@dataclass
class WakeWordConfig:
    wake_words: list = field(default_factory=lambda: ["alexa", "hey assistant"])


AUDIO_CONFIG     = AudioConfig()
WAKE_WORD_CONFIG = WakeWordConfig()