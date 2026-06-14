"""Text-to-speech runtime package."""

__all__ = ["TTSAudioProcessorService", "create_service"]


def __getattr__(name: str):
    if name in {"TTSAudioProcessorService", "create_service"}:
        from .tts_processor import TTSAudioProcessorService, create_service

        return {
            "TTSAudioProcessorService": TTSAudioProcessorService,
            "create_service": create_service,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
