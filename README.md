# Intelligent Web Information Retrieval Agent

[![AMD AI League](https://img.shields.io/badge/AMD%20AI%20League-Match%204-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://amd.com)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An autonomous, **accuracy-first** AI web retrieval agent built for **AMD AI League Match 4**. Given a starting website URL and a natural-language query, the agent autonomously navigates the site, locates the requested factual answer, and provides verbatim supporting proof with the exact source URL.

If the requested information is absent or unverifiable, the agent reliably outputs `found: false` instead of hallucinating.

---

## 🌟 Key Features

- **AMD ROCm & Instinct GPU Ready:** Connects to any OpenAI-compatible API (local vLLM on AMD ROCm/Instinct GPUs, or cloud endpoints) configured solely via `.env` variables—**zero code changes needed**.
- **Accuracy-First Double Guardrails:**
  1. **Deterministic Code Check:** Verifies the extracted evidence quote is an exact, un-tampered substring of the fetched raw page text.
  2. **Secondary LLM Verification:** Checks factual entailment to ensure the quote genuinely answers the query.
  3. **Self-Correction & Backtracking:** Rejects hallucinated quotes automatically and continues exploring alternative candidate links.
- **Hallucination Resistance:** Explicitly recognizes impossible or absent queries and responds with `found: false`.
- **BM25-Powered Efficiency:** Ranks page chunks with BM25Okapi to fit long pages into context and ranks links to select the top 25 most promising navigation paths.
- **Enterprise Safety & SSRF Guard:**
  - Prevents SSRF attacks against RFC1918 private subnets, loopback, and cloud metadata (`169.254.169.254`).
  - Strict same-domain confinement prevents agent drift.
  - Asynchronous robots.txt compliance and per-host rate limiting.
- **Modular Adapters:**
  - **Interactive Web UI:** Modern glassmorphic interface with live Server-Sent Events (SSE) reasoning streaming.
  - **REST API:** High-performance async FastAPI backend with streaming and synchronous endpoints.
  - **Batch Task Processor:** Processes task files in JSON array or JSONL format.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    User([User / Competition Benchmark]) --> Adapters

    subgraph Adapters ["Modular Adapters Layer (src/adapters)"]
        UI[Glassmorphic Web UI]
        API[FastAPI REST & SSE Stream]
        CLI[Batch File Processor]
    end

    Adapters --> Orchestrator

    subgraph Agent ["Agent Core (src/agent)"]
        Orchestrator[Agent Orchestrator & State Machine]
        PromptBuilder[Prompt Builder & Untrusted Data Wrapper]
        ActionParser[Pydantic Action Parser]
        Verifier[Dual-Stage Accuracy Verifier]
        Orchestrator --> PromptBuilder
        PromptBuilder --> ActionParser
        ActionParser --> Orchestrator
        Orchestrator --> Verifier
    end

    subgraph Tools ["Tools & Safety (src/tools & src/safety)"]
        URLGuard[SSRF & URL Guard]
        RobotsLimiter[Robots.txt & Host Rate Limiter]
        Fetcher[Async HTTP Fetcher (httpx)]
        Extractor[Trafilatura & Selectolax Cleaner]
        BM25Ranker[BM25 Chunk & Link Ranker]

        URLGuard --> RobotsLimiter --> Fetcher --> Extractor --> BM25Ranker
    end

    subgraph LLM ["LLM Client (src/llm)"]
        LLMClient[OpenAI-Compatible Client]
        ThoughtStripper[Reasoning / Thought Tag Stripper]
        LLMClient --> ThoughtStripper
    end

    subgraph Hardware ["AMD Hardware / Backend"]
        AMDServer["AMD ROCm / vLLM Server\n(MI210 / MI250 / MI300X / Radeon)"]
        CloudEndpoint["Cloud OpenAI-Compatible Endpoint\n(Gemini / GPT / Mistral / DeepSeek)"]
    end

    Orchestrator --> Tools
    Tools --> Orchestrator
    Orchestrator --> LLMClient
    Verifier --> LLMClient
    LLMClient --> AMDServer
    LLMClient --> CloudEndpoint
```

---

## 🚀 AMD ROCm & Hardware Acceleration

This agent is architected for seamless deployment on **AMD Instinct™ GPUs (MI210, MI250, MI300X)** and **Radeon™ GPUs** running [vLLM with ROCm](https://docs.vllm.ai/en/latest/getting_started/amd-installation.html).

### 1. Launching vLLM on AMD ROCm

Run your model on AMD hardware using the official ROCm Docker container:

```bash
docker run -it --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  rocm/vllm:latest \
  vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype float16
```

### 2. Connect the Agent (Zero Code Changes)

In your `.env` file, point the agent to your AMD server:

```env
LLM_BASE_URL=http://localhost:8000/v1
LLM_API_KEY=EMPTY
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

The agent uses standard OpenAI chat completions, dynamically stripping any chain-of-thought `<thought>` blocks or markdown formatting emitted by reasoning models.

---

## ⚡ Quickstart

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/Joshua8976-spec/Web-Retrieval-Agent.git
cd Web-Retrieval-Agent

python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your API configuration:

```bash
cp .env.example .env
```

Example configuration for testing with Google Gemini's OpenAI-compatible endpoint:
```env
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_API_KEY=your_gemini_api_key_here
LLM_MODEL=gemini-flash-lite-latest
```

### 4. Verify LLM Connection

```bash
python -m src.llm.client
```

Output:
```
[INFO] Attempting connection to https://generativelanguage.googleapis.com/v1beta/openai/...
[SUCCESS] LLM connection validated! Model 'gemini-flash-lite-latest' is ready.
```

### 5. Launch the Web UI & API Server

```bash
uvicorn src.adapters.http_api:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🖥️ Web Interface

The built-in web UI provides a dark glassmorphic dashboard featuring:
- **Live SSE Navigation Stream:** Watch the agent formulate thoughts, inspect links, and visit sub-pages in real time.
- **Verified Proof Card:** Highlights the exact verbatim evidence sentence extracted from the webpage.
- **Direct Source URL:** Clickable link directly to the page containing the evidence.
- **Direct Metric Badges:** Step count, execution duration, and confidence indicator.

---

## 🔌 API & Adapter Usage

### REST Endpoint: `POST /api/retrieve`

```bash
curl -X POST http://localhost:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://fastapi.tiangolo.com/",
    "query": "Who is the original author of FastAPI?"
  }'
```

**Response:**
```json
{
  "found": true,
  "answer": "FastAPI was created by Sebastián Ramírez (tiangolo).",
  "evidence": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python created by Sebastián Ramírez.",
  "source_url": "https://fastapi.tiangolo.com/",
  "steps_taken": 1,
  "pages_visited": ["https://fastapi.tiangolo.com/"],
  "history": [
    {
      "step": 1,
      "url": "https://fastapi.tiangolo.com/",
      "thought": "The homepage clearly introduces the creator in the header section.",
      "action": "answer"
    }
  ]
}
```

### Streaming SSE Endpoint: `GET /api/retrieve/stream`

Connect any event stream client to inspect steps in real time:

```bash
curl -N "http://localhost:8000/api/retrieve/stream?url=https://docs.python.org/3/&query=What+is+Python"
```

Events emitted:
- `event: step` (step index, thought, chosen action)
- `event: complete` (final JSON output with evidence and verification)
- `event: error` (error details if aborted)

### Batch Task File Adapter

Run bulk evaluations or competition batches from a `.json` or `.jsonl` file:

```bash
python -m src.adapters.task_file eval/questions.jsonl -o eval/batch_results.jsonl
```

---

## 📊 Evaluation Harness & Benchmark

The project includes an evaluation suite with 28 curated questions across three difficulty tiers:

| Tier | Questions | Description |
|---|---|---|
| **Easy** | 10 | Direct facts located on the starting URL (1 step). |
| **Medium** | 10 | Facts requiring multi-hop link navigation across the website (2–5 steps). |
| **Impossible** | 8 | Questions with answers not present on the site; tests hallucination prevention (`found: false`). |

### Running the Evaluation

```bash
python eval/run_eval.py
```

Options:
- `--limit N`: Test only the first N questions.
- `--concurrency C`: Set concurrent worker tasks (default: 1).
- `--output PATH`: Specify custom report destination (default: `eval/report.json`).

### Benchmark Metrics Evaluated

- **Overall Accuracy:** % of correct answers (`found=true` on positive questions, `found=false` on impossible questions).
- **Hallucination Rate:** % of impossible questions where the agent falsely claimed to find an answer.
- **Accuracy by Difficulty:** Granular accuracy across Easy, Medium, and Impossible categories.
- **Average Steps:** Mean number of navigation decisions per query.
- **Average Latency:** Mean retrieval duration in seconds.

---

## 🛡️ Safety & Guardrails

| Threat | Mitigation Mechanism | Implementation |
|---|---|---|
| **SSRF / Cloud Metadata Theft** | Blocks private IP ranges (10.x, 192.168.x, 172.16.x), loopback (`127.0.0.1`, `localhost`), and metadata service (`169.254.169.254`). | [src/safety/url_guard.py](file:///c:/Users/joshu/Myspace/Hackathon/Web%20Retrieval%20Agent/src/safety/url_guard.py) |
| **Domain Escaping** | Enforces strict same-domain confinement (`urlparse.netloc`). Subdomain hops and external links are rejected. | [src/tools/links.py](file:///c:/Users/joshu/Myspace/Hackathon/Web%20Retrieval%20Agent/src/tools/links.py) |
| **Server Overload** | Per-host polite rate limiting (default 0.5s delay) and `robots.txt` respect. | [src/safety/robots.py](file:///c:/Users/joshu/Myspace/Hackathon/Web%20Retrieval%20Agent/src/safety/robots.py) |
| **Resource Exhaustion** | Strict 5 MB response size cap; binary mime-type rejection (`image/*`, `video/*`, `application/pdf`). | [src/tools/fetch.py](file:///c:/Users/joshu/Myspace/Hackathon/Web%20Retrieval%20Agent/src/tools/fetch.py) |
| **Runaway Loops** | Circuit breakers: 10 page limit, 120s max timeout, and candidate link deduplication queue. | [src/agent/state.py](file:///c:/Users/joshu/Myspace/Hackathon/Web%20Retrieval%20Agent/src/agent/state.py) |
| **Prompt Injection** | Web content is treated as untrusted 3rd-party data, wrapped in strict structural delimiters. | [src/agent/prompts.py](file:///c:/Users/joshu/Myspace/Hackathon/Web%20Retrieval%20Agent/src/agent/prompts.py) |

---

## 🐳 Docker Deployment

Build and run the containerized agent with a single command:

```bash
# Build image
docker build -t web-retrieval-agent:latest .

# Run container (injecting environment variables)
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name web-retrieval-agent \
  web-retrieval-agent:latest
```

Check health:
```bash
curl http://localhost:8000/api/health
```

---

## 🧪 Testing

The test suite includes 59 automated unit and integration tests covering all agent components:

```bash
pytest -v
```

Test breakdown:
- `test_actions.py`: Pydantic validation of agent action schemas.
- `test_adapters.py`: REST API endpoints, streaming SSE, and batch file I/O.
- `test_config.py`: Environment configuration and secret masking.
- `test_eval.py`: Benchmark questions dataset integrity and metric calculations.
- `test_extract.py`: Trafilatura extraction, chunking, and BM25 ranking.
- `test_fetch.py`: Async HTTP fetching, redirect handling, and binary content filtering.
- `test_links.py`: Link extraction, normalization, and BM25 link relevance ranking.
- `test_llm_client.py`: OpenAI API communication and `<thought>` token stripping.
- `test_loop.py`: Orchestrator multi-step navigation, verification, and backtracking.
- `test_safety.py`: SSRF prevention, IP filtering, and robots.txt cache.
- `test_state.py`: Agent state transitions, circuit breakers, and history tracking.
- `test_verifier.py`: Substring match validation, whitespace normalization, and LLM verification.

---

## 📂 Project Structure

```
Web-Retrieval-Agent/
├── .env.example              # Sample environment configuration
├── Dockerfile                # Production-ready Docker container
├── requirements.txt          # Python dependencies
├── README.md                 # Complete documentation
├── docs/                     # Match specifications and build brief
├── src/
│   ├── config.py             # Settings with pydantic-settings
│   ├── agent/                # Agent core
│   │   ├── actions.py        # Action models (open_link, open_url, answer, give_up)
│   │   ├── loop.py           # Orchestrator & multi-step navigation loop
│   │   ├── prompts.py        # System and step prompts
│   │   ├── state.py          # State tracking & circuit breakers
│   │   └── verifier.py       # Dual-stage accuracy verifier
│   ├── adapters/             # Modular I/O interfaces
│   │   ├── http_api.py       # FastAPI REST API + SSE stream
│   │   └── task_file.py      # JSON / JSONL batch processor
│   ├── llm/
│   │   └── client.py         # OpenAI-compatible client with thought-tag stripping
│   ├── safety/               # Hardening & defense
│   │   ├── robots.py         # Robots.txt cache & rate limiter
│   │   └── url_guard.py      # SSRF & private IP validator
│   └── tools/                # Web tools
│       ├── extract.py        # Trafilatura & BM25 text chunker
│       ├── fetch.py          # Async HTTP client with size/mime caps
│       └── links.py          # URL normalizer & BM25 link ranker
├── eval/                     # Evaluation benchmark
│   ├── questions.jsonl       # 28 benchmark questions (Easy, Medium, Impossible)
│   └── run_eval.py           # Automated evaluation runner
├── tests/                    # 59 automated tests
└── ui/                       # Glassmorphic web interface
    └── index.html            # Single-page interface with live SSE streaming
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
