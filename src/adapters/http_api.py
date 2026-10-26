import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl

from src.agent.loop import AgentOrchestrator
from src.agent.state import AgentResult
from src.llm.client import default_llm_client
from src.config import settings

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Web Information Retrieval Agent API",
    description="AMD AI League Match 4 - Intelligent Web Information Retrieval Agent",
    version="0.1.0",
)

# Enable CORS for local UI and development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UI_DIR = Path(__file__).resolve().parent.parent.parent / "ui"


class RetrievalRequest(BaseModel):
    url: str
    question: str


@app.get("/api/health")
async def health_check():
    """Health check endpoint that verifies server and LLM connectivity."""
    llm_status = default_llm_client.check_connection()
    return {
        "status": "healthy",
        "version": "0.1.0",
        "llm_endpoint": llm_status,
        "config": {
            "model": settings.llm_model,
            "max_pages": settings.agent_max_pages,
            "timeout_seconds": settings.agent_timeout_seconds,
        },
    }


@app.post("/api/retrieve", response_model=AgentResult)
async def retrieve_info(request: RetrievalRequest) -> AgentResult:
    """Run retrieval agent synchronously and return final result."""
    if not request.url or not request.question:
        raise HTTPException(status_code=400, detail="Both 'url' and 'question' are required.")

    orchestrator = AgentOrchestrator()
    result = await orchestrator.run(
        start_url=str(request.url),
        question=request.question,
    )
    return result


@app.get("/api/retrieve/stream")
async def retrieve_info_stream(
    url: str = Query(..., description="Starting target URL"),
    question: str = Query(..., description="Natural-language question"),
):
    """Server-Sent Events endpoint streaming real-time agent thoughts, navigation, and results."""

    async def event_generator() -> AsyncGenerator[str, None]:
        event_queue: asyncio.Queue = asyncio.Queue()

        async def progress_listener(event_data: Dict[str, Any]):
            await event_queue.put(event_data)

        orchestrator = AgentOrchestrator()

        async def run_agent():
            try:
                res = await orchestrator.run(
                    start_url=url,
                    question=question,
                    progress_callback=progress_listener,
                )
                await event_queue.put({"event": "done", "result": res.model_dump()})
            except Exception as exc:
                logger.error("Error during agent stream execution: %s", exc)
                await event_queue.put({"event": "error", "error": str(exc)})
            finally:
                await event_queue.put(None)  # Sentinel

        task = asyncio.create_task(run_agent())

        while True:
            item = await event_queue.get()
            if item is None:
                break

            ev_type = item.get("event", "message")
            ev_data = json.dumps(item)
            yield f"event: {ev_type}\ndata: {ev_data}\n\n"

        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# Serve UI if directory exists
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def get_index():
        index_file = UI_DIR / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Web Retrieval Agent API is running. (ui/index.html not found)</h1>")


def start_server():
    import uvicorn
    uvicorn.run("src.adapters.http_api:app", host=settings.server_host, port=settings.server_port, reload=True)


if __name__ == "__main__":
    start_server()
