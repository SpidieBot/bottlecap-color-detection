# Dockerfile — FINAL VERSION (works 100%)
FROM python:3.11-slim AS base
WORKDIR /app

# Install Poetry
RUN pip install poetry && \
    poetry config virtualenvs.create false

# Copy only dependency files first (for better caching)
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only main --no-interaction --no-ansi

COPY bsort/ ./bsort/
COPY configs/ ./configs/
COPY dataset/ ./dataset/

# Install the project in editable mode
RUN poetry install --only main

# Final image
FROM python:3.11-slim
WORKDIR /app
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin
COPY bsort/ ./bsort/
COPY configs/ ./configs/
COPY dataset/ ./dataset/

# Entry point
ENTRYPOINT ["bsort"]