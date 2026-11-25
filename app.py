"""Gradio web interface for the shared hardsub2srt processing pipeline."""

from __future__ import annotations

import os
import hashlib
import queue
import shutil
import threading
from pathlib import Path
from typing import Iterator

import gradio as gr
import yaml

from core.llm_refiner import LLMConfig
from core.ocr_engine import OCREngine
from core.pipeline import BatchSummary, PipelineConfig, ProgressEvent, run_batch
from core.roi_detector import ROI, detect_roi
from core.srt_builder import BuilderConfig

ROOT = Path(__file__).resolve().parent
WEB_CACHE = ROOT / "web_cache"
DEFAULTS = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
VIDEO_TYPES = [".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm"]


def _public_copy(path: Path) -> str:
    """Expose generated files through a narrow Gradio allowlist."""
    WEB_CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    target = WEB_CACHE / f"{digest}_{path.name}"
    shutil.copy2(path, target)
    return str(target)


def _tool(name: str) -> str:
    configured = str(DEFAULTS.get("video", {}).get(name, name))
    if shutil.which(configured) or Path(configured).is_file():
        return configured
    local = ROOT / ".tools" / "ffmpeg"
    if local.is_dir():
        match = next(local.rglob(name + ".exe"), None)
        if match:
            return str(match)
    return configured


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _source(path_text: str, uploaded: str | None) -> Path:
    if uploaded:
        path = Path(uploaded).resolve()
    elif path_text.strip():
        path = _path(path_text.strip())
    else:
        raise ValueError("Hãy nhập đường dẫn video/thư mục hoặc tải lên một video.")
    if not path.exists():
        raise ValueError(f"Không tìm thấy đầu vào: {path}")
    return path


def _roi(auto: bool, x_min: float, y_min: float, x_max: float, y_max: float) -> ROI | None:
    return None if auto else ROI(x_min, y_min, x_max, y_max)


def _config(
    path_text: str, uploaded: str | None, output_text: str, language: str, fps: int,
    recursive: bool, preserve: bool, overwrite: bool, auto_roi: bool,
    x_min: float, y_min: float, x_max: float, y_max: float,
    watermark_text: str, llm_enabled: bool, llm_base_url: str,
    llm_api_key: str, llm_model: str,
) -> PipelineConfig:
    if not output_text.strip():
        raise ValueError("Hãy nhập thư mục đầu ra.")
    roi_cfg = DEFAULTS.get("roi", {})
    srt_cfg = DEFAULTS.get("srt", {})
    llm_cfg = DEFAULTS.get("llm", {})
    terms = tuple(part.strip() for part in watermark_text.split(",") if part.strip())
    return PipelineConfig(
        input_path=_source(path_text, uploaded), output_dir=_path(output_text.strip()),
        language=language, fps=float(fps), recursive=recursive,
        preserve_subfolders=preserve, overwrite=overwrite, auto_roi=auto_roi,
        manual_roi=_roi(auto_roi, x_min, y_min, x_max, y_max),
        roi_sample_frames=roi_cfg.get("sample_frames", 30),
        roi_max_sample_frames=roi_cfg.get("max_sample_frames", 90),
        roi_probe_max_width=roi_cfg.get("probe_max_width", 960),
        roi_minimum_support_frames=roi_cfg.get("minimum_support_frames", 2),
        roi_padding_ratio=roi_cfg.get("padding_ratio", 0.03),
        prefer_gpu=DEFAULTS.get("ocr", {}).get("prefer_gpu", True),
        ocr_model_size=DEFAULTS.get("ocr", {}).get("model_size", "small"),
        builder=BuilderConfig(
            similarity_threshold=srt_cfg.get("similarity_threshold", 85),
            watermark_seconds=srt_cfg.get("watermark_seconds", 15),
            minimum_duration_seconds=srt_cfg.get("minimum_duration_seconds", 0.15),
            bridge_gap_seconds=srt_cfg.get("bridge_gap_seconds", 0.12),
            minimum_samples=srt_cfg.get("minimum_samples", 2),
            watermark_terms=terms,
        ),
        llm=LLMConfig(
            enabled=llm_enabled, base_url=llm_base_url.strip() or None,
            api_key=llm_api_key.strip() or os.getenv("HARDSUB_LLM_API_KEY") or None,
            model=llm_model.strip() or None,
            batch_size=llm_cfg.get("batch_size", 30),
            timeout_seconds=llm_cfg.get("timeout_seconds", 90),
        ),
        ffmpeg=_tool("ffmpeg"), ffprobe=_tool("ffprobe"),
    )


def preview_roi(
    path_text: str, uploaded: str | None, output_text: str, auto_roi: bool,
    x_min: float, y_min: float, x_max: float, y_max: float,
) -> tuple[str | None, str]:
    """Render one preview before committing to full video OCR."""
    try:
        source = _source(path_text, uploaded)
        if not source.is_file():
            raise ValueError("Ảnh xem trước cần một tệp video, không phải thư mục.")
        output = _path(output_text.strip() or "output")
        manual = _roi(auto_roi, x_min, y_min, x_max, y_max)
        engine = OCREngine(
            prefer_gpu=DEFAULTS.get("ocr", {}).get("prefer_gpu", True),
            model_size=DEFAULTS.get("ocr", {}).get("model_size", "small"),
        ) if auto_roi else None
        roi_cfg = DEFAULTS.get("roi", {})
        result = detect_roi(
            source, engine, manual=manual,
            preview_path=output / f"{source.stem}_roi_preview.jpg",
            sample_frames=roi_cfg.get("sample_frames", 30),
            max_sample_frames=roi_cfg.get("max_sample_frames", 90),
            probe_max_width=roi_cfg.get("probe_max_width", 960),
            minimum_support_frames=roi_cfg.get("minimum_support_frames", 2),
            padding_ratio=roi_cfg.get("padding_ratio", 0.03),
            ffmpeg=_tool("ffmpeg"), ffprobe=_tool("ffprobe"),
        )
        return _public_copy(result.preview_path), f"{result.message} ROI: {result.roi}"
    except Exception as exc:
        return None, f"Lỗi xem trước ROI: {exc}"


def run_ui(
    path_text: str, uploaded: str | None, output_text: str, language: str, fps: int,
    recursive: bool, preserve: bool, overwrite: bool, auto_roi: bool,
    x_min: float, y_min: float, x_max: float, y_max: float,
    watermark_text: str, llm_enabled: bool, llm_base_url: str,
    llm_api_key: str, llm_model: str, progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple[str, str, str | None, str | None]]:
    """Stream live events from the blocking pipeline into Gradio outputs."""
    try:
        config = _config(path_text, uploaded, output_text, language, fps, recursive,
                         preserve, overwrite, auto_roi, x_min, y_min, x_max, y_max,
                         watermark_text, llm_enabled, llm_base_url, llm_api_key, llm_model)
    except Exception as exc:
        yield f"Lỗi cấu hình: {exc}", str(exc), None, None
        return

    events: queue.Queue[tuple[str, object]] = queue.Queue()

    def worker() -> None:
        try:
            result = run_batch(config, callback=lambda event: events.put(("event", event)))
            events.put(("summary", result))
        except Exception as exc:
            events.put(("error", exc))
        finally:
            events.put(("end", None))

    threading.Thread(target=worker, daemon=True).start()
    logs = [f"Đầu vào: {config.input_path}", f"Đầu ra: {config.output_dir}"]
    image_path: str | None = None
    source_preview: Path | None = None
    download_path: str | None = None
    status = "Đang chuẩn bị..."
    yield status, "\n".join(logs), image_path, download_path
    while True:
        kind, value = events.get()
        if kind == "end":
            break
        if kind == "event":
            event = value
            assert isinstance(event, ProgressEvent)
            if event.phase == "roi":
                status = f"Đang tìm ROI: {event.video.name}"
                logs.append(status)
                progress(0, desc=status)
            elif event.phase == "ocr":
                if event.preview_path and event.preview_path.is_file() and event.preview_path != source_preview:
                    source_preview = event.preview_path
                    image_path = _public_copy(event.preview_path)
                if event.processed_frames:
                    fraction = min(1.0, event.processed_frames / max(1, event.duration_seconds * config.fps))
                    status = (f"Đang OCR: {event.video.name} — {event.processed_frames} khung hình, "
                              f"{event.processing_fps:.1f} FPS")
                    progress(fraction, desc=status)
                    if event.processed_frames % 30 == 0:
                        logs.append(status)
                else:
                    status = f"Đã chọn ROI: {event.roi}"
                    logs.append(f"{event.video.name}: {event.message} {event.roi}")
            elif event.phase == "done":
                status = f"Hoàn thành {event.video.name}: {event.message}"
                logs.append(status)
                progress(1, desc=status)
            elif event.phase == "failed":
                status = f"Lỗi {event.video.name}: {event.message}"
                logs.append(status)
        elif kind == "summary":
            summary = value
            assert isinstance(summary, BatchSummary)
            for item in summary.files:
                logs.append(f"{item.status}: {item.video.name} → {item.output} ({item.message})")
            status = (f"Tổng kết: {summary.completed} hoàn thành, {summary.skipped} bỏ qua, "
                      f"{summary.failed} lỗi")
            if len(summary.files) == 1 and summary.files[0].output.is_file():
                download_path = _public_copy(summary.files[0].output)
        else:
            status = f"Lỗi xử lý: {value}"
            logs.append(status)
        yield status, "\n".join(logs[-60:]), image_path, download_path


def build_app() -> gr.Blocks:
    WEB_CACHE.mkdir(parents=True, exist_ok=True)
    default_output = os.getenv("HARDSUB_OUTPUT_DIR", str(ROOT / "output"))
    default_input = os.getenv("HARDSUB_INPUT_DIR", "")
    with gr.Blocks(title="Hardsub2srt") as demo:
        gr.Markdown("# Hardsub2srt\nTrích phụ đề cứng từ video thành tệp SRT.")
        with gr.Row():
            input_path = gr.Textbox(value=default_input, label="Đường dẫn video hoặc thư mục trên máy chủ")
            upload = gr.File(label="Hoặc tải lên một video", file_types=VIDEO_TYPES, type="filepath")
        output_dir = gr.Textbox(value=default_output, label="Thư mục đầu ra")
        with gr.Row():
            language = gr.Dropdown(choices=[("Tự động / đa ngôn ngữ", "auto"), ("Tiếng Trung", "ch"),
                                            ("Tiếng Việt", "vi"), ("Tiếng Nhật", "ja"),
                                            ("Tiếng Anh", "en")], value="auto", label="Ngôn ngữ OCR")
            fps = gr.Slider(1, 10, value=3, step=1, label="Khung hình mỗi giây")
        with gr.Row():
            recursive = gr.Checkbox(value=False, label="Quét thư mục con")
            preserve = gr.Checkbox(value=False, label="Giữ cấu trúc thư mục con")
            overwrite = gr.Checkbox(value=False, label="Ghi đè SRT đã có")
        with gr.Accordion("Vùng phụ đề (ROI)", open=True):
            auto_roi = gr.Checkbox(value=True, label="Tự động phát hiện ROI")
            gr.Markdown("Tắt tự động để dùng bốn tỷ lệ ROI thủ công (0–1).")
            with gr.Row():
                x_min = gr.Slider(0, 1, value=0, step=0.01, label="X nhỏ nhất")
                y_min = gr.Slider(0, 1, value=0.65, step=0.01, label="Y nhỏ nhất")
                x_max = gr.Slider(0, 1, value=1, step=0.01, label="X lớn nhất")
                y_max = gr.Slider(0, 1, value=1, step=0.01, label="Y lớn nhất")
            preview_button = gr.Button("Xem trước ROI")
            preview_image = gr.Image(label="Ảnh xem trước ROI", interactive=False, height=420)
        watermark_terms = gr.Textbox(value=", ".join(DEFAULTS.get("srt", {}).get("watermark_terms", [])),
                                    label="Từ khóa watermark cần bỏ qua (ngăn cách bằng dấu phẩy)",
                                    placeholder="Ví dụ: tên logo")
        with gr.Accordion("Sửa lỗi OCR bằng LLM (tùy chọn)", open=False):
            llm_enabled = gr.Checkbox(value=False, label="Bật sửa OCR bằng LLM")
            llm_base_url = gr.Textbox(label="Base URL", placeholder="http://localhost:11434/v1")
            llm_api_key = gr.Textbox(label="API Key", type="password")
            llm_model = gr.Textbox(label="Tên mô hình")
        run_button = gr.Button("Bắt đầu xử lý", variant="primary")
        status = gr.Textbox(label="Trạng thái", interactive=False)
        logs = gr.Textbox(label="Nhật ký trực tiếp", lines=12, interactive=False)
        download = gr.File(label="Tải tệp SRT (chạy một video)", interactive=False)

        preview_button.click(
            preview_roi,
            inputs=[input_path, upload, output_dir, auto_roi, x_min, y_min, x_max, y_max],
            outputs=[preview_image, status],
        )
        run_button.click(
            run_ui,
            inputs=[input_path, upload, output_dir, language, fps, recursive, preserve,
                    overwrite, auto_roi, x_min, y_min, x_max, y_max, watermark_terms,
                    llm_enabled, llm_base_url, llm_api_key, llm_model],
            outputs=[status, logs, preview_image, download],
        )
    return demo


if __name__ == "__main__":
    user, password = os.getenv("HARDSUB_WEB_USER"), os.getenv("HARDSUB_WEB_PASSWORD")
    if bool(user) != bool(password):
        raise SystemExit("Phải đặt cả HARDSUB_WEB_USER và HARDSUB_WEB_PASSWORD.")
    auth = (user, password) if user and password else None
    local_windows = os.name == "nt"
    server_name = os.getenv("HARDSUB_WEB_HOST") or ("127.0.0.1" if local_windows else "0.0.0.0")
    build_app().queue(default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=7860, inbrowser=local_windows, auth=auth,
        allowed_paths=[str(WEB_CACHE)], show_error=True,
    )
