"""Speech-to-text runtime package."""

__all__ = ["STTAudioProcessorService", "create_service"]


def __getattr__(name: str):
    if name in {"STTAudioProcessorService", "create_service"}:
        from .stt_processor import STTAudioProcessorService, create_service

        return {
            "STTAudioProcessorService": STTAudioProcessorService,
            "create_service": create_service,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
