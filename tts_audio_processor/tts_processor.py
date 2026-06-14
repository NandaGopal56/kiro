import asyncio

from communication_bus.inmemory_bus import InMemoryBus, bus as default_bus

from .logger import logger
from .text_reader import make_llm_response_handler


class TTSAudioProcessorService:
    """Service for managing TTS lifecycle and message handling."""

    name = "tts_audio_processor"

    def __init__(
        self,
        bus: InMemoryBus | None = None,
        synthesizer=None,
        player=None,
    ):
        self._owns_bus = bus is None
        self.bus: InMemoryBus = bus or default_bus
        self.synthesizer = synthesizer
        self.player = player
        self._is_running = False
        self._run_task: asyncio.Task | None = None
        self._handler = None

    async def _run(self):
        """Main TTS loop (extend later for TTS processing)."""
        while self._is_running:
            await asyncio.sleep(0.5)  # placeholder loop
            # could add periodic work here if needed
        logger.info("TTS loop stopped")

    async def start(self, **kwargs) -> None:
        """Start the TTS service asynchronously."""
        if self._is_running:
            logger.warning("TTS service already running")
            return

        try:
            logger.info("Starting TTS Service...")
            await self.bus.connect()
            if self.player is None:
                from .audio_player import TTSPlayer

                self.player = TTSPlayer()
            if hasattr(self.player, "start"):
                self.player.start()
            self._handler = make_llm_response_handler(
                synthesizer=self.synthesizer,
                player=self.player,
            )
            self.bus.subscribe("voice/commands/llm_response", self._handler)

            self._is_running = True
            # self._run_task = asyncio.create_task(self._run())
            logger.info("TTS Service started")

        except Exception as e:
            logger.error(f"Failed to start TTS service: {e}", exc_info=True)
            self._is_running = False
            if self._owns_bus:
                await self.bus.disconnect()
            raise

    async def stop(self) -> None:
        """Stop the TTS service and clean up resources."""
        if not self._is_running:
            return

        logger.info("Stopping TTS Service...")
        self._is_running = False
        if self._handler is not None:
            self.bus.unsubscribe("voice/commands/llm_response", self._handler)
            self._handler = None
        if self.player is not None and hasattr(self.player, "stop"):
            self.player.stop()

        if self._run_task:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass

        if self._owns_bus:
            await self.bus.disconnect()
        logger.info("TTS Service stopped")


def create_service(
    bus: InMemoryBus | None = None,
    synthesizer=None,
    player=None,
) -> TTSAudioProcessorService:
    return TTSAudioProcessorService(
        bus=bus,
        synthesizer=synthesizer,
        player=player,
    )
