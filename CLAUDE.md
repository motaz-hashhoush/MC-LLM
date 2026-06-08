# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Run the stack
```bash
docker compose -f docker/docker-compose.yml up -d
curl http://localhost:8080/health
```

The compose file launches **Redis**, **vLLM**, and the **FastAPI gateway**, but **does NOT** launch PostgreSQL. PostgreSQL must already be running on the host — the `fastapi` container reaches it via `host.docker.internal` (see `extra_hosts` in `docker/docker-compose.yml`).

### Initialise PostgreSQL (one-time, on the host)
```bash
psql -U postgres -f docker/init_db.sql
```
Creates the `llm_user` role, `llm_logs` database, and both `request_logs` and `tts_requests` tables. `DatabaseManager.init_db()` also calls `Base.metadata.create_all` at startup, so the SQL script is mostly for fresh hosts / explicit grants.

### Download model weights (required before first `up`)
```bash
python scripts/download_model.py --model Qwen/Qwen3-14B-AWQ --dir models
```
Downloads the LLM **and** SILMA TTS into `./models/`, which is bind-mounted into both the vLLM and FastAPI containers at `/models`. Use `--tts-only` or `--skip-tts` to control which. STT (faster-whisper) is downloaded the same way using its repo ID (`deepdml/faster-whisper-large-v3-turbo-ct2`) but is not yet wired into `download_model.py` — fetch it manually with `huggingface_hub` if missing from `./models/deepdml/...`.

### Local dev without Docker
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```
Reads `.env` from the repo root. You still need a running Redis, PostgreSQL, and vLLM endpoint reachable at the configured URLs.

### Database utilities
```bash
python scripts/test_db.py        # verify DATABASE_URL connectivity
python scripts/export_db.py      # pg_dump of llm_logs
python scripts/import_db.py      # restore from dump
```

### Tests
There is **no test suite** in this repo. Don't invent a pytest command — verify behaviour by hitting the live endpoints (`/summarize`, `/rewrite`, `/generate`, `/stt`, `/v1/tts`, `/health`).

## Architecture

### Request flow
```
HTTP → FastAPI route → TaskProcessor.process()
                         │
                         ├─ JobQueue.enqueue_job()  ──► Redis LIST  "llm:tasks"
                         │                                     │
                         │                                     ▼ (in-process background task)
                         │                              QueueConsumer._consume_loop
                         │                                     │
                         │                                     ├─ DBLogger.log_request()
                         │                                     ├─ InferenceEngine.process_task()
                         │                                     │      └─ VLLMClient → vLLM /v1/chat/completions
                         │                                     ├─ JobQueue.store_result()
                         │                                     └─ DBLogger.log_completion/log_error()
                         │
                         └─ poll  job:{id}:status every 500 ms until COMPLETED/FAILED or JOB_TIMEOUT
```

**Critical detail**: `QueueConsumer` is **not a separate worker process** — it runs as an asyncio background task started in `lifespan()` in [app/main.py](app/main.py). Single FastAPI process, single consumer, single in-flight job (the queue is FIFO but processing is serial because the consumer awaits one `process_task` at a time). The Redis queue exists mainly for decoupling and crash-safety, not horizontal scaling. If you need parallel inference, you'd add multiple `QueueConsumer` instances or scale out via separate workers.

**STT and TTS bypass the queue entirely** — `/stt` calls `WhisperClient.transcribe()` directly on `app.state.whisper_client`, and `/v1/tts` calls `TTSProcessor.process_tts_job()` directly. Both offload CPU/GPU work to threads via `asyncio.to_thread`.

### Redis key layout (see [app/queue/job_queue.py](app/queue/job_queue.py))
- `llm:tasks` — LIST, `LPUSH` for enqueue / `BRPOP` for dequeue
- `job:{id}:status` — STRING, one of `pending|processing|completed|failed`
- `job:{id}:payload` — STRING, full job JSON
- `job:{id}:result` — STRING, `{"result": ..., "error": ...}`, TTL = `RESULT_TTL` (default 3600s)

### Task-type specifics
The three LLM endpoints share `_handle_task()` but differ in two important ways:

1. **Prompt templates** ([app/llm/prompts.py](app/llm/prompts.py)) are applied for `summarize` and `rewrite`. For `generate` the user's `prompt` is sent verbatim as a single user message — there is no template. Add a `_Template` to `TEMPLATES` to wire in a new templated task.

2. **`think` flag** is only honoured for `/generate`; `/summarize` and `/rewrite` force it to `False` regardless of what the client sends. When think is off, `InferenceEngine` appends `/no_think` to the prompt (Qwen-specific control token) **and** strips any leftover `<think>...</think>` tags from the response. See [app/services/inference_engine.py:48-70](app/services/inference_engine.py#L48-L70).

### Startup model loading
`lifespan()` loads Whisper STT and SILMA TTS **concurrently** via `asyncio.gather(asyncio.to_thread(WhisperClient), asyncio.to_thread(silma_client.load))`. Both are CPU-bound model loads — running them serially adds ~1–2 minutes to startup.

SILMA TTS has a **warmup step** in `SilmaTTSClient.load()` ([app/tts/silma_client.py:130-155](app/tts/silma_client.py#L130-L155)): it runs one synthesis against the bundled Arabic reference audio with the exact hardcoded transcript. Without this, the very first user request would block 1–2 min while the library auto-transcribes the reference audio via its internal Whisper model. **Do not remove the warmup or change the hardcoded `_DEFAULT_REF_TEXT`** — it's a literal-match cache key.

### TTS voice cloning
SILMA TTS is a zero-shot model and **requires** a reference audio file. When the user supplies `clone_audio` (base64 WAV), `TTSProcessor` writes it to a temp file and passes it to `synthesize()` with empty `ref_text` (forces re-transcription, cached after first use). When omitted, `synthesize()` falls back to the bundled `/usr/local/lib/python3.12/site-packages/silma_tts/infer/ref_audio_samples/ar.ref.24k.wav` with the exact transcript to hit the warmup cache. That hardcoded site-packages path means SILMA must be installed at the exact Python 3.12 path used in the Dockerfile.

### NFE step count
Both `load()` warmup and `synthesize()` pick `nfe_step = 16 if "cuda" in self.device else 8`. This is a quality/speed tradeoff coupled to the device — change both call sites together or extract to a config setting.

### Config
All settings live in [app/config.py](app/config.py) as a `pydantic-settings` `Settings` class, cached via `@lru_cache`. Settings load from environment variables (and `.env` at repo root). Container env vars are set in `docker-compose.yml`. There are two layers: the `.env` file feeds the compose file (`${VAR}` interpolation), which then sets container env vars, which Pydantic reads.

`STT_MODEL_PATH` has a subtle behaviour: if set (non-empty), `WhisperClient` loads from that local directory; if empty string, it falls back to HuggingFace hub download of `STT_MODEL_NAME` at runtime. See [app/stt/whisper_client.py:42-43](app/stt/whisper_client.py#L42-L43).

## Gotchas

- **Python 3.12 only**: the Dockerfile installs Python 3.12 from the deadsnakes PPA. The SILMA TTS reference-audio path is hardcoded against `/usr/local/lib/python3.12/site-packages/...`, so a Python version bump requires updating those literal paths too.
- **CUDA 12.1 lock**: `requirements.txt` pins torch to `2.3.1+cu121` from the PyTorch CUDA 12.1 index. The base image is `nvidia/cuda:12.1.0-devel-ubuntu22.04`. Don't bump these independently.
- **vLLM is not in `requirements.txt`** — it runs from the prebuilt `vllm/vllm-openai:latest` image. The FastAPI container only needs an HTTP client to reach it.
- **Job timeout vs HTTP timeout**: `TaskProcessor` polls for `JOB_TIMEOUT` seconds (default 120) and then returns a `FAILED` `TaskResponse` to the client. The job may still complete in Redis after the response is sent — there's no cancellation propagated to the consumer.
- **Empty `result` is treated as success**: `TaskResponse` returns `status=COMPLETED` even if `result_data.get("result")` is `None`, as long as no error key was stored. If the model returns an empty string, the client gets `status=completed, result=""`.
