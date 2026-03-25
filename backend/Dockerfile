FROM python:3.11-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy the dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies system-wide
RUN uv sync --no-dev

# Copy the actual application code
COPY src/ ./src/

# Expose the FastAPI port
EXPOSE 8000

# Run the production server
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
