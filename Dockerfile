FROM flwr/base:1.20.0-py3.11-ubuntu24.04

WORKDIR /app
COPY pyproject.toml .
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
    && python -m pip install -U --no-cache-dir .

COPY data data/