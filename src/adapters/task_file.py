import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.agent.loop import AgentOrchestrator
from src.agent.state import AgentResult

logger = logging.getLogger(__name__)


class TaskItem(BaseModel):
    """Input task specification."""
    id: Optional[str] = Field(default=None, description="Unique task identifier.")
    url: str = Field(description="Starting target URL.")
    question: str = Field(description="Natural-language question.")


class TaskOutputItem(BaseModel):
    """Output result matching Match 4 format."""
    id: Optional[str] = None
    found: bool
    status: str = "not_found_on_site"
    answer: Optional[str] = None
    evidence_quote: Optional[str] = None
    source_url: Optional[str] = None
    explanation: Optional[str] = None
    pages_visited: List[str] = Field(default_factory=list)
    steps: int = 0
    elapsed_seconds: float = 0.0


def load_tasks(file_path: Path) -> List[TaskItem]:
    """Load tasks from either a standard JSON array file or JSON Lines (.jsonl)."""
    tasks: List[TaskItem] = []
    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    if file_path.suffix.lower() == ".jsonl":
        for idx, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if line:
                data = json.loads(line)
                if "id" not in data or not data["id"]:
                    data["id"] = f"task_{idx}"
                tasks.append(TaskItem.model_validate(data))
    else:
        # Standard JSON array or single object
        raw_data = json.loads(content)
        if isinstance(raw_data, list):
            for idx, item in enumerate(raw_data, 1):
                if "id" not in item or not item["id"]:
                    item["id"] = f"task_{idx}"
                tasks.append(TaskItem.model_validate(item))
        elif isinstance(raw_data, dict):
            if "id" not in raw_data or not raw_data["id"]:
                raw_data["id"] = "task_1"
            tasks.append(TaskItem.model_validate(raw_data))

    return tasks


async def run_task(orchestrator: AgentOrchestrator, task: TaskItem) -> TaskOutputItem:
    """Run agent on single task and format output."""
    res = await orchestrator.run(start_url=task.url, question=task.question)
    return TaskOutputItem(
        id=task.id,
        found=res.found,
        status=res.status,
        answer=res.answer,
        evidence_quote=res.evidence_quote,
        source_url=res.source_url,
        explanation=res.explanation,
        pages_visited=res.pages_visited,
        steps=res.steps,
        elapsed_seconds=res.elapsed_seconds,
    )


async def process_task_file(
    input_path: Path,
    output_path: Path,
    orchestrator: Optional[AgentOrchestrator] = None,
) -> List[TaskOutputItem]:
    """Execute all tasks from input file and save formatted results to output file."""
    tasks = load_tasks(input_path)
    logger.info("Loaded %d tasks from %s", len(tasks), input_path)

    orch = orchestrator or AgentOrchestrator()
    results: List[TaskOutputItem] = []

    for idx, task in enumerate(tasks, 1):
        logger.info("[%d/%d] Processing task %s: '%s'", idx, len(tasks), task.id, task.question)
        out = await run_task(orch, task)
        results.append(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)

    logger.info("Saved %d task results to %s", len(results), output_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Match 4 Task File Adapter")
    parser.add_argument("--input", "-i", required=True, help="Path to input tasks JSON or JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Path to output results JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(process_task_file(Path(args.input), Path(args.output)))


if __name__ == "__main__":
    main()
