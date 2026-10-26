# Multi-stage or slim production container for Web Retrieval Agent
FROM python:3.12-slim

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install curl for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies first for efficient layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create an unprivileged user to run the container securely
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --gid 1001 appuser

# Copy application code and UI assets
COPY src/ ./src/
COPY ui/ ./ui/
COPY eval/ ./eval/
COPY docs/ ./docs/

# Fix permissions for non-root execution
RUN chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Container healthcheck against FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch FastAPI ASGI server
CMD ["uvicorn", "src.adapters.http_api:app", "--host", "0.0.0.0", "--port", "8000"]
