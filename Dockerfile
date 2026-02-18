FROM python:3.12-slim

RUN pip install --no-cache-dir uv==0.5.8

WORKDIR /app

# Install dependencies first (separate layer for caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --no-cache

# Copy source
COPY . .

CMD uv run uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}
