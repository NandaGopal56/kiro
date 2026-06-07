import asyncio
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from communication_bus.inmemory_bus import bus
from agents.bot import invoke_conversation
from .logger import logger
from pathlib import Path


class UIService:
    def __init__(self):
        self.app = FastAPI()
        self.bus = bus
        self._is_running = False

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

        from agents.bot import invoke_conversation

        async def event_gen():
            try:
                async for part in invoke_conversation(message, thread_id=thread_id):
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
            uvicorn.Config(self.app, host="localhost", port=8000, log_level="info")
        )

        asyncio.create_task(self._server.serve())
        logger.info("UI Service running at http://localhost:8000")

    async def stop(self):
        if not self._is_running:
            return
        self._is_running = False
        await self.bus.disconnect()


async def main():
    ui = UIService()
    await ui.start()
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
