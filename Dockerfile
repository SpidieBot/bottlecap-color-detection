FROM python:3.11-slim AS builder
WORKDIR /app

# Install system deps + Poetry
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
COPY bsort ./bsort
COPY configs ./configs

# Install deps + project
RUN poetry install --no-interaction --no-ansi && \
    rm -rf /root/.cache/pypoetry /root/.cache/pip 

# Final image
FROM python:3.11-slim
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy source (minimal)
COPY bsort ./bsort
COPY configs ./configs

# Entry point
ENTRYPOINT ["bsort"]