# Intelligent Web Information Retrieval Agent

**Prepared for AMD AI League - Match 4**

An autonomous, accuracy-first AI web retrieval agent. Given a website URL and a natural-language question, it navigates the site, locates the requested facts, and returns verbatim proof with direct source links.

---

## Key Features
- **AMD Hardware Compatible:** Connects to any OpenAI-compatible API (e.g. local vLLM running on AMD ROCm/Instinct GPUs, or hosted endpoints) configured solely via environment variables.
- **Accuracy-First Architecture:** 
  - Substring code-check to guarantee verbatim quotes.
  - Secondary LLM verification pass.
  - Returns `{"found": false}` instead of hallucinating.
- **Efficient Navigation:** BM25 chunk ranking for long pages and top-25 link relevance filtering before navigation.
- **Safety & SSRF Guard:** Enforces private IP blocking, domain confinement, and rate limiting.

---

## Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Environment Setup
Copy `.env.example` to `.env` and configure your LLM endpoint:
```bash
cp .env.example .env
```

### 3. Verify LLM Connectivity
```bash
python -m src.llm.client
```

### 4. Run Tests
```bash
pytest -v
```
