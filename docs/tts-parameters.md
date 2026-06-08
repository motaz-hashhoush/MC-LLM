# Supertonic 3 TTS — Control Parameters

**Endpoint:** `POST /v1/tts`  
**Content-Type:** `application/json`  
**Response:** Raw audio bytes (`audio/wav` or `audio/mpeg`)

---

## Parameters

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `text` | string | — | 1–10 000 chars | Text to synthesise. Long inputs are auto-chunked with 0.3 s silence between chunks. |
| `language` | string | `"ar"` | See table below | ISO 639-1 language code, or `"na"` for automatic detection. |
| `voice_name` | string | `"M1"` | M1–M5, F1–F5 | Voice identity. M = male, F = female. All voices support all languages. |
| `speed` | float | `1.05` | 0.7 – 2.0 | Playback speed multiplier. `1.0` = natural pace. |
| `total_steps` | int | `8` | 5 – 12 | Denoising steps. More steps = higher quality but slower synthesis. |
| `format` | string | `"wav"` | `"wav"`, `"mp3"` | Output audio format. WAV is 16-bit PCM at 44 100 Hz. MP3 requires ffmpeg. |

---

## Voices

| ID | Gender |
|----|--------|
| M1 | Male |
| M2 | Male |
| M3 | Male |
| M4 | Male |
| M5 | Male |
| F1 | Female |
| F2 | Female |
| F3 | Female |
| F4 | Female |
| F5 | Female |

---

## Supported Languages

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `ar` | Arabic | `fr` | French | `ru` | Russian |
| `bg` | Bulgarian | `hi` | Hindi | `sk` | Slovak |
| `cs` | Czech | `hr` | Croatian | `sl` | Slovenian |
| `da` | Danish | `hu` | Hungarian | `sv` | Swedish |
| `de` | German | `id` | Indonesian | `tr` | Turkish |
| `el` | Greek | `it` | Italian | `uk` | Ukrainian |
| `en` | English | `ja` | Japanese | `vi` | Vietnamese |
| `es` | Spanish | `ko` | Korean | `zh` | Chinese |
| `et` | Estonian | `lt` | Lithuanian | `na` | Auto-detect |
| `fi` | Finnish | `lv` | Latvian | | |
| `nl` | Dutch | `pl` | Polish | | |
| `pt` | Portuguese | `ro` | Romanian | | |

---

## `total_steps` — Quality vs Speed Guide

| Steps | Quality | CPU Synthesis Time (approx.) |
|-------|---------|------------------------------|
| 5 | Fast / acceptable | ~0.5 × real-time |
| 8 | Balanced (default) | ~1.0 × real-time |
| 12 | Highest quality | ~1.5 × real-time |

> The container uses CPU-only inference (ONNX Runtime, `CPUExecutionProvider`).  
> Synthesis time scales roughly linearly with `total_steps`.

---

## Example Requests

### Minimal (Arabic, all defaults)
```bash
curl -X POST http://localhost:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "مرحبا كيف حالك"}' \
  --output output.wav
```

### Female voice, slow speed, MP3
```bash
curl -X POST http://localhost:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "مرحبا كيف حالك",
    "language": "ar",
    "voice_name": "F2",
    "speed": 0.9,
    "total_steps": 10,
    "format": "mp3"
  }' \
  --output output.mp3
```

### English, high quality
```bash
curl -X POST http://localhost:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test.",
    "language": "en",
    "voice_name": "M3",
    "speed": 1.0,
    "total_steps": 12,
    "format": "wav"
  }' \
  --output output.wav
```

### Auto language detection
```bash
curl -X POST http://localhost:8080/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bonjour le monde",
    "language": "na",
    "voice_name": "F1"
  }' \
  --output output.wav
```

---

## Notes

- **Max text length:** 10 000 characters per request. Split longer text into multiple requests.
- **Concurrent requests:** Synthesis is serialised internally (one request at a time) to prevent ONNX session collisions. Requests queue automatically.
- **First request after restart:** The model performs a warmup synthesis at startup, so the first real request will not incur extra delay.
- **MP3 output** requires `ffmpeg` to be installed in the container (it is, by default).
