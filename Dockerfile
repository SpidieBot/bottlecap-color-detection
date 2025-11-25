FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi && \
    rm -rf /root/.cache/pypoetry  # Clean cache to free space

COPY bsort ./bsort
COPY configs ./configs

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local /usr/local
COPY bsort ./bsort
COPY configs ./configs

ENTRYPOINT ["bsort"]