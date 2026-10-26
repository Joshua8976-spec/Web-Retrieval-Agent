# AMD AI League - Match 4: Intelligent Web Information Retrieval
## Official Competition Specification & Unknowns Resolution

> **Instructions for User:** Please paste the official Match 4 requirements into this document as soon as they are released. The core agent architecture remains modular and decoupled, with all I/O isolated under `src/adapters/` so adaptation takes minutes.

---

### 1. Input & Output Specifications
- **Input Format:** *(e.g., CLI arguments, JSON task file, HTTP API endpoint, environment variables)*
  - Details: 
- **Output Format:** *(e.g., JSON file on disk, stdout stream, HTTP response body, evaluation schema)*
  - Details: 

---

### 2. Scoring & Evaluation Criteria
- **Scoring Metric:** *(e.g., Exact match on quote, fuzzy match, F1 score, penalty for hallucination, time/step penalties)*
  - Details:
- **Ground Truth Format:** *(Expected fields: answer, evidence_quote, source_url, etc.)*
  - Details:

---

### 3. Model & Runtime Environment
- **Target Model / Endpoint:** *(e.g., vLLM endpoint hosted on AMD ROCm/Instinct GPU, local open model, external API)*
  - Details:
- **Network Access During Grading:** *(Is full internet access allowed, or are tasks evaluated against local mirrors / mock web servers / static datasets?)*
  - Details:

---

### 4. Packaging & Submission
- **Docker Requirements:** *(Base image, AMD ROCm base vs CPU lightweight container, entrypoint format)*
  - Details:
- **Resource Constraints:** *(Memory, CPU/GPU, execution timeout per query)*
  - Details:

---

### 5. Additional Rules / Edge Cases
- *(Any specific restrictions on external libraries, robots.txt enforcement, rate limits, concurrent runs, or user-agent strings)*
