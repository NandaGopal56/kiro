import asyncio
from typing import Any, Callable, Dict, List, Optional, Protocol

from .logger import logger


class MessageBus(Protocol):
    async def start(self, **kwargs) -> None:
        ...

    async def stop(self) -> None:
        ...

    async def connect(self) -> None:
        ...

    async def disconnect(self) -> None:
        ...

    async def publish(self, topic: str, payload: Any) -> None:
        ...

    def subscribe(self, topic: str, callback: Callable[[str, Any], Any]) -> None:
        ...

    def unsubscribe(
        self,
        topic: str,
        callback: Optional[Callable[[str, Any], Any]] = None,
    ) -> None:
        ...


class InMemoryBus:
    """
    Lightweight in-memory message bus for pub/sub communication.
    Works without any external broker.
    """

    name = "communication_bus"

    def __init__(self):
        self.callbacks: Dict[str, List[Callable[[str, Any], Any]]] = {}
        self.queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self.running = False
        self.task: Optional[asyncio.Task] = None

    async def _loop(self):
        """Internal loop to dispatch messages from the queue."""
        while self.running:
            try:
                topic, payload = await asyncio.wait_for(self.queue.get(), timeout=0.5)
                if topic in self.callbacks:
                    for callback in self.callbacks[topic]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(topic, payload)
                            else:
                                callback(topic, payload)
                        except Exception as e:
                            logger.error(f"Error in callback for {topic}: {e}")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in bus loop: {e}")

    async def connect(self):
        """Start the bus loop."""
        if not self.running:
            self.running = True
            self.task = asyncio.create_task(self._loop())
            logger.info("InMemoryBus started")

    async def start(self, **kwargs) -> None:
        await self.connect()

    async def disconnect(self):
        """Stop the bus loop."""
        self.running = False
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.task = None
        logger.info("InMemoryBus stopped")

    async def stop(self) -> None:
        await self.disconnect()

    async def publish(self, topic: str, payload: Any):
        """Publish a message to a topic."""
        await self.queue.put((topic, payload))

    def subscribe(self, topic: str, callback: Callable[[str, Any], Any]):
        """Subscribe a callback to a topic.
        
        Args:
            topic: Topic to subscribe to
            callback: Can be a regular function or a coroutine function
        """
        if topic not in self.callbacks:
            self.callbacks[topic] = []
        if callback not in self.callbacks[topic]:
            self.callbacks[topic].append(callback)
            logger.info(f"Subscribed to {topic}")

    def unsubscribe(
        self,
        topic: str,
        callback: Optional[Callable[[str, Any], Any]] = None,
    ):
        """Unsubscribe from a topic or remove a specific callback.
        
        Args:
            topic: Topic to unsubscribe from
            callback: Optional specific callback to remove. If None, removes all callbacks for the topic.
        """
        if topic in self.callbacks:
            if callback is None:
                del self.callbacks[topic]
                logger.info(f"Unsubscribed from {topic}")
            else:
                if callback in self.callbacks[topic]:
                    self.callbacks[topic].remove(callback)
                    logger.debug(f"Removed callback for {topic}")
                if not self.callbacks[topic]:
                    del self.callbacks[topic]
                    logger.info(f"Unsubscribed from {topic} (no more callbacks)")


def create_bus() -> InMemoryBus:
    return InMemoryBus()


bus = InMemoryBus()
