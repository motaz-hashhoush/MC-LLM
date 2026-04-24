# 🔮 MC-LLM — Production LLM Inference Infrastructure

A modular, production-ready system for serving LLM inference over REST. Built with **FastAPI**, **vLLM**, **Redis**, and **PostgreSQL**.

---

## Architecture

```
Client Applications
        │
        ▼
  FastAPI Gateway  (:8080)
        │
        ├──▶  Redis Queue
        │         │
        │         ▼
        │    Queue Consumer
        │         │
        │         ▼
        │    Inference Engine
        │         │
        │         ▼
        │    vLLM Server (:8000)  ──▶  GPU
        │
        └──▶  PostgreSQL (request logs)
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose v2
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

### Run

```bash
docker compose -f docker/docker-compose.yml up -d
```

Verify the gateway is running:

```bash
curl http://localhost:8080/health
```

---

## API Reference

### `POST /summarize`

Summarise a block of text.

```bash
curl -X POST http://localhost:8080/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article text here...", "max_tokens": 200, "temperature": 0.7}'
```

### `POST /rewrite`

Professionally rewrite text.

```bash
curl -X POST http://localhost:8080/rewrite \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here...", "max_tokens": 300, "temperature": 0.5}'
```

### `POST /generate`

Generate a detailed article.

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Topic or outline...", "max_tokens": 512, "temperature": 0.8}'
```

### `GET /health`

Health check endpoint.

```json
{"status": "ok", "version": "1.0.0", "model": "Qwen/Qwen3-14B"}
```

---

## Configuration

All settings are controlled via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen3-14B` | HuggingFace model ID |
| `VLLM_ENDPOINT` | `http://vllm:8000` | vLLM server URL |
| `REDIS_HOST` | `redis` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `DATABASE_URL` | `postgresql+asyncpg://llm_user:llm_pass@postgres:5432/llm_logs` | PostgreSQL connection string |
| `MAX_TOKENS` | `512` | Default max generation tokens |
| `TEMPERATURE` | `0.7` | Default sampling temperature |

---

## Project Structure

```
MC_LLM/
├── app/
│   ├── main.py              # FastAPI entry point + lifespan
│   ├── config.py            # Pydantic settings
│   ├── api/
│   │   ├── routes.py        # REST endpoints
│   │   └── schemas.py       # Pydantic models
│   ├── queue/
│   │   ├── redis_client.py  # Async Redis connection
│   │   ├── job_queue.py     # Job queue operations
│   │   └── queue_consumer.py # Background worker
│   ├── llm/
│   │   ├── vllm_client.py   # vLLM API wrapper
│   │   └── prompts.py       # Task prompt templates
│   ├── db/
│   │   ├── models.py        # SQLAlchemy ORM models
│   │   ├── database.py      # Async engine management
│   │   └── db_logger.py     # Structured DB logging
│   └── services/
│       ├── task_processor.py # Request orchestrator
│       └── inference_engine.py # Direct inference logic
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Changing the Model

1. Update `MODEL_NAME` in your environment / `.env` file.
2. Update the `--model` argument in `docker-compose.yml` under the `vllm` service.
3. Restart the stack: `docker compose -f docker/docker-compose.yml up -d`.

---

## License

MIT