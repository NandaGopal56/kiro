import asyncio
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from communication_bus.inmemory_bus import bus as default_bus

from .logger import logger


class UIService:
    name = "live_interaction"

    def __init__(
        self,
        bus=None,
        agent_client: Callable | None = None,
        host: str = "localhost",
        port: int = 8000,
    ):
        self.app = FastAPI()
        self._owns_bus = bus is None
        self.bus = bus or default_bus
        self.agent_client = agent_client or _default_agent_client
        self.host = host
        self.port = port
        self._is_running = False
        self._server = None
        self._server_task: asyncio.Task | None = None

        self.app.get("/")(self.index)
        self.app.websocket("/ws/camera")(self.camera_ws)
        self.app.get("/api/chat/stream")(self.chat_stream)

    async def index(self):
        base_dir = Path(__file__).parent
        face_html_path = base_dir / "face.html"

        if not face_html_path.exists():
            logger.error(f"face.html not found at {face_html_path}")
            return HTMLResponse(
                "<h1>Error: UI template not found</h1>",
                status_code=500,
            )
        return HTMLResponse(face_html_path.read_text(encoding="utf-8"))

    async def camera_ws(self, websocket: WebSocket):
        await websocket.accept()
        logger.info("Camera WS connected")

        try:
            while True:
                frame = await websocket.receive_bytes()
                await self.bus.publish("camera/front", frame)
        except Exception:
            logger.info("Camera WS disconnected")

    # Removed non-streaming chat endpoint to avoid duplication

    async def chat_stream(self, request: Request):
        """
        Streaming chat via Server-Sent Events (SSE).
        GET /api/chat/stream?message=...
        """
        message = request.query_params.get("message", "")
        thread_id = int(request.query_params.get("thread_id", "1") or 1)
        if not message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        async def event_gen():
            try:
                async for part in self.agent_client(message, thread_id=thread_id):
                    if part:
                        yield f"data: {part}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
            except Exception as e:
                logger.exception("Chat stream failed")
                yield f"event: error\ndata: {str(e)}\n\n"

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)

    async def start(self):
        if self._is_running:
            return

        await self.bus.connect()
        self._is_running = True

        import uvicorn
        self._server = uvicorn.Server(
            uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        )

        self._server_task = asyncio.create_task(self._server.serve())
        logger.info(f"UI Service running at http://{self.host}:{self.port}")

    async def stop(self):
        if not self._is_running:
            return
        self._is_running = False
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            try:
                await asyncio.wait_for(self._server_task, timeout=3)
            except asyncio.TimeoutError:
                self._server_task.cancel()
            self._server_task = None
        if self._owns_bus:
            await self.bus.disconnect()


async def _default_agent_client(message: str, thread_id: int = 1):
    from agents.bot import invoke_conversation

    async for part in invoke_conversation(message, thread_id=thread_id):
        yield part


def create_service(
    bus=None,
    agent_client: Callable | None = None,
    host: str = "localhost",
    port: int = 8000,
) -> UIService:
    return UIService(
        bus=bus,
        agent_client=agent_client,
        host=host,
        port=port,
    )


async def main():
    ui = UIService()
    await ui.start()
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
