import asyncio
from collections.abc import Callable
from typing import Optional, Protocol

from communication_bus.inmemory_bus import InMemoryBus, bus as default_bus

from .utils.logger import logger


class Assistant(Protocol):
    async def run(self) -> None:
        ...

    async def stop(self) -> None:
        ...


class STTAudioProcessorService:
    """Service to manage the voice assistant and its communication with the message bus."""

    name = "stt_audio_processor"
    
    def __init__(
        self,
        bus: InMemoryBus | None = None,
        assistant: Assistant | None = None,
        assistant_factory: Callable[[Callable[[str], None]], Assistant] | None = None,
    ):
        """Initialize the audio processor service."""
        self._owns_bus = bus is None
        self.bus: InMemoryBus = bus or default_bus
        self._assistant_factory = assistant_factory or _create_live_assistant
        self.assistant = assistant
        self._is_running = False
        self._run_task: Optional[asyncio.Task] = None

    def _publish_utterance(self, utterance: str) -> None:
        asyncio.get_running_loop().create_task(
            self.bus.publish(
                "voice/commands",
                {"text": utterance, "source": "stt"},
            )
        )
    
    async def start(self, **kwargs) -> None:
        """Start the audio processor service asynchronously."""
        if self._is_running:
            logger.warning("Audio processor is already running")
            return
            
        try:
            # Connect to the message bus
            await self.bus.connect()

            if self.assistant is None:
                self.assistant = self._assistant_factory(self._publish_utterance)
            
            # Start the assistant in a separate task
            self._is_running = True
            self._run_task = asyncio.create_task(self._run())
            
        except Exception as e:
            logger.error(f"Failed to start audio processor: {e}", exc_info=True)
            self._is_running = False
            if self._owns_bus:
                await self.bus.disconnect()
            raise
    
    async def _run(self) -> None:
        """Run the main processing loop."""
        try:
            if self.assistant:
                await self.assistant.run()
        except asyncio.CancelledError as e:
            logger.error(f"Audio processor run task cancelled: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error in audio processor run task: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop the audio processor service asynchronously."""
        if not self._is_running:
            return
            
        try:
            if self.assistant:
                await self.assistant.stop()
            if self._owns_bus:
                await self.bus.disconnect()
            
            if self._run_task:
                self._run_task.cancel()
                try:
                    await self._run_task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            logger.error(f"Error stopping audio processor: {e}", exc_info=True)
            raise
        finally:
            self._is_running = False


def _create_live_assistant(on_utterance: Callable[[str], None]) -> Assistant:
    from .src.voice_assistant import VoiceProcessor

    return VoiceProcessor(on_utterance=on_utterance)


def create_service(
    bus: InMemoryBus | None = None,
    provider=None,
    audio_handler=None,
    assistant: Assistant | None = None,
) -> STTAudioProcessorService:
    if assistant is not None:
        return STTAudioProcessorService(bus=bus, assistant=assistant)

    def factory(on_utterance: Callable[[str], None]) -> Assistant:
        from .src.voice_assistant import VoiceProcessor

        return VoiceProcessor(
            provider=provider,
            audio_handler=audio_handler,
            on_utterance=on_utterance,
        )

    return STTAudioProcessorService(bus=bus, assistant_factory=factory)
