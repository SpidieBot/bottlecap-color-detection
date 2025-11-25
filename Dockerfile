# Dockerfile — FINAL WORKING VERSION (Dec 2025)
FROM python:3.11-slim AS base
WORKDIR /app

# Install system deps + Poetry
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

# Copy EVERYTHING needed for installation FIRST
COPY pyproject.toml poetry.lock ./
COPY bsort ./bsort
COPY configs ./configs
COPY dataset/ ./dataset/

# Now install (bsort folder exists → no more error)
RUN poetry install --only main --no-interaction --no-ansi

# Final minimal image
FROM python:3.11-slim
WORKDIR /app

# Copy installed packages
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy source code
COPY bsort ./bsort
COPY configs ./configs
COPY dataset/ ./dataset/

# Entry point
ENTRYPOINT ["bsort"]