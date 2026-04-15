FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

# Copy monorepo root files
COPY pyproject.toml uv.lock README.md ./

# Copy packages and apps
COPY packages/fedotllm ./packages/fedotllm
COPY packages/shared ./packages/shared
COPY apps/frontend ./apps/frontend
COPY apps/server ./apps/server

RUN uv sync --frozen --no-dev && uv cache clean

EXPOSE 8501 8000

# Default to frontend (Streamlit)
CMD ["streamlit", "run", "apps/frontend/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
