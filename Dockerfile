FROM python:3.11-slim

# git is required for repo ingestion.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY cli ./cli
COPY examples ./examples

ENV DATA_DIR=/app/data \
    REPO_WORKDIR=/app/data/repos \
    PYTHONUNBUFFERED=1

EXPOSE 8000
VOLUME ["/app/data"]

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
