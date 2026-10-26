import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import AsyncClient, ASGITransport

from src.adapters.task_file import load_tasks, process_task_file, TaskItem
from src.adapters.http_api import app
from src.agent.loop import AgentOrchestrator
from src.agent.state import AgentResult


@pytest.fixture
def tmp_tasks_json(tmp_path):
    tasks_file = tmp_path / "tasks.json"
    data = [
        {"id": "t1", "url": "https://example.com", "question": "What is Python?"},
        {"id": "t2", "url": "https://example.com/rocm", "question": "What is ROCm?"},
    ]
    tasks_file.write_text(json.dumps(data), encoding="utf-8")
    return tasks_file


@pytest.fixture
def tmp_tasks_jsonl(tmp_path):
    tasks_file = tmp_path / "tasks.jsonl"
    lines = [
        json.dumps({"id": "t1", "url": "https://example.com", "question": "What is Python?"}),
        json.dumps({"id": "t2", "url": "https://example.com/rocm", "question": "What is ROCm?"}),
    ]
    tasks_file.write_text("\n".join(lines), encoding="utf-8")
    return tasks_file


def test_load_tasks_json_array(tmp_tasks_json):
    tasks = load_tasks(tmp_tasks_json)
    assert len(tasks) == 2
    assert tasks[0].id == "t1"
    assert tasks[1].url == "https://example.com/rocm"


def test_load_tasks_jsonl(tmp_tasks_jsonl):
    tasks = load_tasks(tmp_tasks_jsonl)
    assert len(tasks) == 2
    assert tasks[0].id == "t1"
    assert tasks[1].question == "What is ROCm?"


@pytest.mark.asyncio
async def test_process_task_file(tmp_tasks_json, tmp_path):
    output_file = tmp_path / "results.json"
    mock_orch = MagicMock()
    mock_orch.run = AsyncMock(
        return_value=AgentResult(
            found=True,
            answer="Sample answer",
            evidence_quote="Sample quote",
            source_url="https://example.com",
            steps=1,
            elapsed_seconds=1.5,
        )
    )

    results = await process_task_file(tmp_tasks_json, output_file, orchestrator=mock_orch)
    assert len(results) == 2
    assert results[0].found is True
    assert results[0].answer == "Sample answer"
    assert output_file.exists()

    saved_data = json.loads(output_file.read_text(encoding="utf-8"))
    assert len(saved_data) == 2
    assert saved_data[0]["id"] == "t1"


@pytest.mark.asyncio
async def test_http_api_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch("src.adapters.http_api.default_llm_client.check_connection", return_value={"status": "connected"}):
            resp = await client.get("/api/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "healthy"
            assert data["llm_endpoint"]["status"] == "connected"


@pytest.mark.asyncio
async def test_http_api_retrieve_post():
    mock_result = AgentResult(
        found=True,
        answer="FastAPI is fast",
        evidence_quote="FastAPI is a fast web framework.",
        source_url="https://fastapi.tiangolo.com",
        steps=2,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with patch.object(AgentOrchestrator, "run", new_callable=AsyncMock, return_value=mock_result):
            resp = await client.post(
                "/api/retrieve",
                json={"url": "https://fastapi.tiangolo.com", "question": "What is FastAPI?"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["found"] is True
            assert data["answer"] == "FastAPI is fast"
