# 1. Base Image
FROM python:3.13-slim

# 2. Create a work directory in container
WORKDIR /app

# 3. Install UV Package Manager
RUN pip install --no-cache-dir uv

# 4. Copy Dependency file
COPY pyproject.toml .

# 5. Create venv and install dependencies
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install .

# 6. Copy Remaining Files
COPY main.py .
COPY app/ app/
COPY docs/ docs/

# 7. Expose the port
EXPOSE 8000

# 8. Run the application
CMD ["/app/.venv/bin/uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]

