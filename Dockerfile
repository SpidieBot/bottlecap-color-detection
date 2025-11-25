FROM python:3.11-slim AS base

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry config virtualenvs.create false

FROM base AS builder
RUN poetry install --only main

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY bsort ./bsort
COPY configs ./configs
COPY dataset ./dataset

ENTRYPOINT ["bsort"]