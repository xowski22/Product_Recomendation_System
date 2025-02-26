FROM ubuntu:latest
LABEL authors="xowski22"

ENTRYPOINT ["top", "-b"]


FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/mappings models/checkpoints

EXPOSE 8080

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]