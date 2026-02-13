FROM python:3.13-slim

WORKDIR /app

# Install uv (fast Python package manager)
RUN pip install --no-cache-dir uv

# Copy dependency file first (Docker caches this layer)
COPY pyproject.toml .

# Install dependencies
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install .

# Copy application code
COPY app/ app/
COPY main.py .
COPY docs/ docs/

# Use the virtual env
ENV PATH="/app/.venv/bin:$PATH"

# Cloud Run sets PORT env var (default 8080)
ENV PORT=8080

# Start the FastAPI server
CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port $PORT"]
