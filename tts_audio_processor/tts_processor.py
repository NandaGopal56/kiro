import asyncio
from communication_bus.inmemory_bus import InMemoryBus, bus
from .text_reader import on_llm_response
from .logger import logger

class TTSAudioProcessorService:
    """Service for managing TTS lifecycle and message handling."""

    def __init__(self):
        self.bus: InMemoryBus = bus
        self._is_running = False
        self._run_task: asyncio.Task | None = None

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
            self.bus.subscribe("voice/commands/llm_response", on_llm_response)

            self._is_running = True
            # self._run_task = asyncio.create_task(self._run())
            logger.info("TTS Service started")

        except Exception as e:
            logger.error(f"Failed to start TTS service: {e}", exc_info=True)
            self._is_running = False
            await self.bus.disconnect()
            raise

    async def stop(self) -> None:
        """Stop the TTS service and clean up resources."""
        if not self._is_running:
            return

        logger.info("Stopping TTS Service...")
        self._is_running = False

        if self._run_task:
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass

        await self.bus.disconnect()
        logger.info("TTS Service stopped")