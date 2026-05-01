# Arabic Speech-to-Text Endpoint

Add a new `POST /transcribe` endpoint to the FastAPI gateway that accepts an audio file upload and returns the Arabic transcription using `openai/whisper-large-v3-turbo` via the Hugging Face `transformers` pipeline.

**Why Whisper over NeMo?**  
- Standard `transformers` library — no heavy NeMo install (~2–3 GB) required.  
- Whisper natively handles long audio (>30 s), multiple formats, and Arabic out-of-the-box.  
- `whisper-large-v3-turbo` is ~800 MB, significantly smaller than the full large-v3.

---

## User Review Required

> [!IMPORTANT]
> The Whisper model loads **in-process** inside the FastAPI container (not via vLLM). The model weights (~800 MB) are downloaded on first startup and cached. After that, startup adds ~10–15 s for GPU loading.

> [!WARNING]
> The `fastapi` service currently has **no GPU reservation** in `docker-compose.yml`. We must add NVIDIA GPU access to the `fastapi` service, or the model will fall back to CPU (slow — ~10–20× slower per request).

> [!CAUTION]
> Both vLLM (Qwen3-14B) and Whisper will share GPU VRAM. Whisper-large-v3-turbo requires ~3–4 GB of VRAM in float16 mode. If the GPU is under memory pressure, consider setting `STT_DEVICE=cpu` in `.env`.

---

## Proposed Changes

### STT Module — `app/stt/`

#### [NEW] `app/stt/__init__.py`
Empty init file.

#### [NEW] `app/stt/whisper_client.py`
Singleton that owns the Whisper pipeline. Responsibilities:
- Load `openai/whisper-large-v3-turbo` at startup via `AutoModelForSpeechSeq2Seq` + `AutoProcessor`.
- Expose `async transcribe(audio_bytes: bytes, filename: str) -> str` that:
  1. Writes bytes to a named temp file (preserving extension for ffmpeg codec detection).
  2. Runs `pipe(tmp_path, generate_kwargs={"language": "arabic", "task": "transcribe"})` in a **thread-pool executor** (the pipeline is synchronous).
  3. Deletes the temp file.
  4. Returns the transcription string.
- Uses `torch.float16` on CUDA, `torch.float32` on CPU.
- Exposes `close()` to release the model from memory on shutdown.

```python
# Conceptual init
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,   # float32 on CPU
    device=device,               # "cuda:0" or "cpu"
)
# Inference
result = pipe(tmp_path, generate_kwargs={"language": "arabic", "task": "transcribe"})
transcript = result["text"]
```

---

### API Layer

#### [MODIFY] `app/api/schemas.py`
Add:
```python
class TranscribeResponse(BaseModel):
    """Response returned by the /transcribe endpoint."""
    id: UUID
    status: JobStatus          # COMPLETED | FAILED
    transcript: str | None = None
    language: str = "ar"
    error: str | None = None
```

#### [MODIFY] `app/api/routes.py`
Add new endpoint using `UploadFile`:
```python
from fastapi import File, UploadFile

@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    request: Request = ...,
) -> TranscribeResponse:
    """Transcribe an uploaded Arabic audio file (WAV / MP3 / M4A / OGG / FLAC)."""
```
- Validates `content_type` and file size against `STT_MAX_AUDIO_MB`.
- Calls `request.app.state.whisper_client.transcribe(...)`.
- Returns `TranscribeResponse`.

---

### App Startup

#### [MODIFY] `app/main.py`
- Import `WhisperClient` from `app.stt.whisper_client`.
- Add to `lifespan`: instantiate, attach as `app.state.whisper_client`, call `.close()` on shutdown.

---

### Configuration

#### [MODIFY] `app/config.py`
```python
# --- STT ---
STT_MODEL_NAME: str = "openai/whisper-large-v3-turbo"
STT_DEVICE: str = "cuda"        # "cuda" | "cpu"
STT_MAX_AUDIO_MB: int = 25
STT_LANGUAGE: str = "arabic"    # force Arabic; set to "" for auto-detect
```

#### [MODIFY] `.env` and `.env.example`
```dotenv
# ── Speech-to-Text ───────────────────────────────────────────────────────────
STT_DEVICE=cuda
STT_MAX_AUDIO_MB=25
STT_LANGUAGE=arabic
```

---

### Dependencies

#### [MODIFY] `requirements.txt`
```
# ── Speech-to-Text (Whisper via Transformers) ────────────────────────────────
transformers>=4.44.0,<5.0.0
accelerate>=0.33.0             # for device_map and efficient loading
torch>=2.3.0                   # already pulled by transformers if CUDA image is used
soundfile>=0.12.1              # fallback audio I/O
python-multipart>=0.0.9        # required by FastAPI UploadFile
```

> [!NOTE]
> `torch` is listed explicitly so pip resolves the CUDA-compatible wheel. If the Docker base image already includes PyTorch (e.g. a `pytorch/pytorch` base), you can remove it.

---

### Docker

#### [MODIFY] `docker/Dockerfile`
Add `libsndfile1` and `ffmpeg` for audio decoding:
```dockerfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*
```

#### [MODIFY] `docker/docker-compose.yml`
Add GPU access to the `fastapi` service (same pattern as `vllm`):
```yaml
fastapi:
  ...
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```
Also pass through `STT_DEVICE`, `STT_MAX_AUDIO_MB`, `STT_LANGUAGE` in the `environment` block.

---

## Open Questions

> [!IMPORTANT]
> **GPU vs CPU**: Do you want Whisper to run on `cuda` (default, fast) or `cpu` (no VRAM cost, ~10–20× slower)? This becomes the default in `.env`.

> [!IMPORTANT]
> **Language mode**: Force Arabic always (`STT_LANGUAGE=arabic`) or allow the caller to pass a `language` query parameter to switch language dynamically?

---

## File Summary

| Action | File |
|---|---|
| **NEW** | `app/stt/__init__.py` |
| **NEW** | `app/stt/whisper_client.py` |
| **MODIFY** | `app/api/schemas.py` |
| **MODIFY** | `app/api/routes.py` |
| **MODIFY** | `app/main.py` |
| **MODIFY** | `app/config.py` |
| **MODIFY** | `.env` + `.env.example` |
| **MODIFY** | `requirements.txt` |
| **MODIFY** | `docker/Dockerfile` |
| **MODIFY** | `docker/docker-compose.yml` |

---

## Verification Plan

### Build & Start
```bash
docker compose -f docker/docker-compose.yml build fastapi
docker compose -f docker/docker-compose.yml up -d
```

### Health Check
```bash
curl http://localhost:8080/health
```

### Transcription Test
```bash
curl -X POST http://localhost:8080/transcribe \
  -F "file=@arabic_sample.wav" \
  | python -m json.tool
```
Verify `transcript` field contains Arabic text and there are no OOM errors in the FastAPI logs.
