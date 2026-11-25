# PROJECT SPECIFICATION: HARDSUB-TO-SRT EXTRACTOR (GPU-ACCELERATED)

## 1. Objective & Core Capabilities
Build a production-ready, GPU-accelerated Python application that extracts hardcoded subtitles (hardsubs) from video files into standard `.srt` subtitle files.
- **Target Languages:** High-accuracy recognition for **Chinese (Simplified & Traditional)**, **Vietnamese (with full diacritics)**, **Japanese**, and English.
- **Deployment Targets:**
  1. Native execution on **Linux VPS (Ubuntu/Debian) with NVIDIA GPU (CUDA)**.
  2. Containerized execution via **Docker & Docker Compose** (`--gpus all` / NVIDIA Container Toolkit) for local PC or cloud deployment.

---

## 2. System Architecture (Decoupled Core + Dual Interface)
Structure the project cleanly into modular packages so the processing engine is completely independent of the UI:

```text
hardsub2srt/
├── core/
│   ├── video_sampler.py      # Fast frame extraction & video metadata reader
│   ├── roi_detector.py       # Automatic Smart ROI subtitle zone detection
│   ├── ocr_engine.py         # GPU OCR wrapper (multi-language)
│   ├── srt_builder.py        # Fuzzy text merging, watermark filtering & SRT formatting
│   ├── llm_refiner.py        # Optional OpenAI-compatible / Ollama post-processor
│   └── pipeline.py           # End-to-end single-file & batch orchestrator
├── cli.py                    # CLI entry point (for headless VPS / tmux batch runs)
├── app.py                    # Lightweight Web UI entry point (Gradio)
├── config.yaml               # Default configuration (paths, thresholds, LLM settings)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 3. Functional Requirements

### A. Input, Output & Batch Processing
- **Input Modes:** Accept either a **single video file** (`.mp4`, `.mkv`, `.avi`, `.mov`, `.ts`, `.webm`) OR an **input directory**.
- **Directory Scanning:** When given a directory, scan and process videos sequentially. Include an option (`--recursive` / UI checkbox) to scan subdirectories.
- **Configurable Output Directory:** Save generated `.srt` files to a configurable `output_dir` (settable via `config.yaml`, CLI flag `--output`, or Web UI input field).
  - Match the `.srt` filename to the source video filename (e.g., `episode_01.mp4` -> `episode_01.srt`).
  - Include an option to preserve relative subfolder structure inside `output_dir` when scanning recursively.
- **Batch Resilience & Resume:**
  - By default, **skip** videos that already have an existing `.srt` file in `output_dir` (allowing safe resume if a batch job is interrupted). Add an `--overwrite` flag / UI toggle to force re-processing.
  - Wrap each file in error handling so one corrupted video logs an error and skips cleanly without crashing the entire batch.

### B. Automatic Smart ROI (Subtitle Region Detection)
To maximize speed (3x–5x faster) and eliminate false positives from background text/signs:
1. **Auto-Detection Mode (Default):**
   - Sample 25–30 frames evenly across the middle 90% of the video (skipping intro/outro credits).
   - Run text detection on sampled frames, cluster bounding boxes by spatial frequency (typically bottom-center, but must support top or vertical layouts), and compute a unified **Smart ROI bounding box** with configurable padding margins.
   - Generate a preview image (`roi_preview.jpg`) drawing the detected green ROI box on a sample frame for visual verification in the Web UI.
2. **Manual Override:** Allow optional manual ROI ratios (`x_min, y_min, x_max, y_max` from `0.0` to `1.0`) in config/CLI/UI in case a specific video has non-standard subtitle placement.

### C. GPU OCR Engine Selection & Optimization
- Select the most reliable, high-accuracy OCR stack that runs cleanly on **NVIDIA GPU (CUDA)** without dependency conflicts in Docker (e.g., **RapidOCR with `onnxruntime-gpu`** using PaddleOCR v4 multi-language models, OR **PaddleOCR GPU** with pinned CUDA/cuDNN versions).
- Explicitly verify CUDA execution provider initialization on startup and log the active GPU device name (e.g., via PyTorch/ONNXRuntime check). Provide graceful fallback to CPU with a clear warning if no GPU is exposed.
- Support language selection (`auto` / `ch` / `vi` / `ja` / `en`). Note: Ensure Vietnamese diacritics and CJK characters are accurately handled by choosing appropriate recognition models.
- **Frame Sampling Rate:** Configurable sampling interval (default: `3` frames per second / every `0.33s`) to balance timestamp accuracy and processing speed. Crop every sampled frame to the Smart ROI before passing to the OCR engine.

### D. Algorithmic Temporal Merging & Filtering
Before outputting `.srt`:
- **Fuzzy Deduplication:** Merge consecutive frames whose recognized text has high similarity (e.g., Levenshtein / `rapidfuzz` ratio >= `85%`) into a single subtitle entry with `start_time` and `end_time`. Pick the highest-confidence OCR text string among the merged frames.
- **Static Watermark / Logo Removal:** Automatically discard text boxes that remain static with identical text for longer than a configurable duration threshold (e.g., `> 15 seconds` continuously).
- **Noise & Flicker Cleanup:** Drop spurious single-frame detections (`duration < 0.15s`) and bridge tiny gaps (`< 0.12s`) between identical consecutive subtitles.

### E. Optional LLM Post-Processing (Toggleable)
- **Disabled by default**; toggleable via CLI (`--llm-refine`) and Web UI checkbox.
- Connect to any **OpenAI-compatible API** (`base_url`, `api_key`, `model`), supporting OpenAI, Gemini (OpenAI endpoint), or local **Ollama** (`http://localhost:11434/v1`).
- Send subtitle entries in batches (e.g., 25–30 lines per request) using structured JSON output keyed by subtitle index `id`.
- **Strict Prompt Constraint:** Instruct the LLM ONLY to fix obvious OCR character errors, broken spacing, and punctuation based on context—**never merge, split, delete, translate, or alter the index IDs or timestamps**.

---

## 4. User Interfaces

### 1. CLI (`cli.py`)
- Build with `typer` or `argparse` + `rich` progress bars.
- Example usage:
  ```bash
  python cli.py --input /data/videos --output /data/subs --lang ch --recursive --fps 3 --llm-refine
  ```
- Display per-file progress bar, current FPS processing speed, detected ROI coordinates, and overall batch summary.

### 2. Web UI (`app.py`)
- Build with **Gradio** (binding to `0.0.0.0:7860` so it works seamlessly over SSH port-forwarding on a VPS or Docker port mapping).
- Include:
  - Input path text box (file or folder path on server/container) + file uploader option for single local files.
  - Output directory path text box.
  - Controls: Language dropdown (`Auto/Multi`, `Chinese`, `Vietnamese`, `Japanese`), Sampling FPS slider (`1` to `10`), Recursive scan checkbox, Overwrite existing `.srt` checkbox.
  - ROI section: Auto-detect toggle + manual ROI sliders + **Image preview** showing the detected Smart ROI box.
  - LLM Post-Processing accordion: Enable toggle, Base URL, API Key, Model name.
  - Live log output box + progress status + Download button for single-file runs.

---

## 5. Docker & VPS Deployment Requirements
- **`Dockerfile`:** Use an official NVIDIA CUDA runtime base image (e.g., `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`), install Python 3.10+, `ffmpeg`, `libgl1`, and all Python dependencies. Pre-download default OCR models during image build so container startup is instant.
- **`docker-compose.yml`:** Configure GPU reservation (`deploy.resources.reservations.devices` with `capabilities: [gpu]`), expose port `7860`, and map volumes `./input:/input`, `./output:/output`, and `./config.yaml:/app/config.yaml`.
- **`README.md`:** Include concise setup instructions for (1) Bare-metal Linux VPS with CUDA and (2) Docker Compose on local PC/VPS.

---

## 6. Implementation Order
Implement and verify in clean stages:
1. Project scaffolding, `config.yaml`, GPU verification utility, and `core/video_sampler.py`.
2. `core/roi_detector.py` (with ROI preview image export) and `core/ocr_engine.py` (GPU-accelerated CJK + Vietnamese OCR).
3. `core/srt_builder.py` (fuzzy merging + watermark filter) and `core/llm_refiner.py`.
4. `core/pipeline.py` + `cli.py` (verify end-to-end batch processing).
5. `app.py` (Gradio Web UI) + `Dockerfile` & `docker-compose.yml`.