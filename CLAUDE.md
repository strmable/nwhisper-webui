# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Whisper-WebUI** is a Gradio-based browser UI for OpenAI's Whisper speech-to-text model. It supports three Whisper implementations (faster-whisper, openai/whisper, insanely-fast-whisper), multiple input sources (file/YouTube/mic), VAD preprocessing, BGM separation, speaker diarization, and subtitle output in SRT/WebVTT/txt/LRC formats. There is also a FastAPI REST backend for async processing.

## Running the App

```bash
# Web UI (Gradio, default port 7860)
python app.py

# With specific whisper implementation
python app.py --whisper_type faster-whisper

# REST API backend (port 8000)
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Docker
docker compose build && docker compose up
```

## Installation

```bash
# Windows
Install.bat

# Unix
bash Install.sh
```

Creates a Python venv and installs from `requirements.txt`. PyTorch is installed from a CUDA-versioned index (default CUDA 12.8); update the `--index-url` in `requirements.txt` for different CUDA versions.

## Running Tests

```bash
# From repo root with venv activated
pytest tests/
pytest backend/tests/
```

Test files cover: `test_transcription.py`, `test_translation.py`, `test_bgm_separation.py`, `test_diarization.py`, `test_vad.py`, `test_config.py`.

## Architecture

### Two-Tier System

**Tier 1 – Gradio Web UI** (`app.py`): Monolithic `App` class that builds the UI and wires callbacks. Five tabs: File, YouTube, Mic, T2T Translation, BGM Separation.

**Tier 2 – FastAPI REST Backend** (`backend/`): Async task-based processing. Client POSTs audio → gets an identifier → polls `/task/{identifier}` for status. Results are cached with TTL-based cleanup.

### Pipeline Flow (both tiers)

1. Input loading (file / YouTube via pytubefix / mic)
2. Preprocessing: VAD (Silero) → BGM separation (UVR)
3. Transcription: chosen Whisper implementation
4. Post-processing: speaker diarization (pyannote)
5. Output: subtitle formatter → SRT/WebVTT/txt/LRC

### Key Modules

| Path | Purpose |
|------|---------|
| `modules/whisper/` | Whisper pipeline; factory + 3 implementations + shared base |
| `modules/whisper/data_classes.py` | Dataclasses for all parameter groups (WhisperParams, VadParams, DiarizationParams, BGMSeparationParams) |
| `modules/vad/silero_vad.py` | Silero VAD – silence filtering before transcription |
| `modules/diarize/` | Speaker diarization via pyannote |
| `modules/translation/` | DeepL API + Facebook NLLB T2T translation |
| `modules/uvr/music_separator.py` | UVR background music/vocal separation |
| `modules/utils/subtitle_manager.py` | SRT, WebVTT, LRC formatting |
| `modules/utils/paths.py` | Central definition of all project directory paths |
| `modules/utils/constants.py` | Global constants and enums |
| `backend/routers/` | FastAPI route handlers (transcription, vad, bgm_separation, task) |
| `backend/db/` | SQLite task persistence via SQLAlchemy/sqlmodel |
| `backend/common/cache_manager.py` | TTL-based cleanup thread for cached results |

### Adding a New Whisper Implementation

Subclass `BaseTranscriptionPipeline` in `modules/whisper/base_transcription_pipeline.py`, register it in `modules/whisper/whisper_factory.py`.

### Configuration

- `configs/default_parameters.yaml` – default model sizes, compute types, VAD/diarization/BGM defaults used by the UI
- `configs/translation.yaml` – i18n strings for 37+ languages
- `backend/configs/config.yaml` – backend model settings, cache TTL and cleanup frequency
- `backend/configs/.env.example` – template for `.env` (HF_TOKEN required for diarization, DB_URL)

## Dependencies

- Python 3.10–3.12 required
- FFmpeg must be installed and on PATH
- CUDA GPU strongly recommended; CPU inference is slow
- `faster-whisper` is the default implementation (≈2.4× more VRAM-efficient than openai/whisper)
- Diarization requires a HuggingFace token (`HF_TOKEN`) to download pyannote models
