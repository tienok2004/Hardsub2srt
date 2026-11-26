"""Gradio web interface for the shared hardsub2srt processing pipeline."""

from __future__ import annotations

import os
import base64
import hashlib
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Iterator

import gradio as gr
import yaml
from PIL import Image

from core.llm_refiner import LLMConfig
from core.ocr_engine import OCREngine
from core.captions import caption_config_from
from core.paths import clean_path_text, pick_local_path
from core.pipeline import BatchSummary, PipelineConfig, ProgressEvent, format_clock, run_batch
from core.roi_detector import ROI, detect_roi
from core.roi_editor import HTML_TEMPLATE, CSS_TEMPLATE, JS_ON_LOAD
from core.srt_builder import BuilderConfig

ROOT = Path(__file__).resolve().parent
WEB_CACHE = ROOT / "web_cache"
DEFAULTS = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
VIDEO_TYPES = [".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm"]
UI_REFRESH_SECONDS = 1.0


def _progress_bar(fraction: float, width: int = 20) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {fraction * 100:5.1f}%"


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


def _clean_text(value: str | None) -> str:
    """Gradio sends None for some empty textboxes."""
    return value.strip() if value is not None else ""


def _source(path_text: str | None, uploaded: str | None) -> Path:
    if uploaded:
        path = Path(uploaded).resolve()
    elif clean_path_text(path_text):
        path = _path(clean_path_text(path_text))
    else:
        raise ValueError("Hãy nhập đường dẫn video/thư mục hoặc tải lên một video.")
    if not path.exists():
        raise ValueError(f"Không tìm thấy đầu vào: {path}")
    return path


def _roi(auto: bool, x_min: float, y_min: float, x_max: float, y_max: float) -> ROI | None:
    return None if auto else ROI(x_min, y_min, x_max, y_max)


def _config(
    path_text: str | None, uploaded: str | None, output_text: str | None, language: str, fps: int,
    recursive: bool, preserve: bool, overwrite: bool, auto_roi: bool,
    x_min: float, y_min: float, x_max: float, y_max: float,
    watermark_text: str | None, llm_enabled: bool, llm_base_url: str | None,
    llm_api_key: str | None, llm_model: str | None, captions_output: str = "merged",
) -> PipelineConfig:
    output_value = clean_path_text(output_text)
    if not output_value:
        raise ValueError("Hãy nhập thư mục đầu ra.")
    roi_cfg = DEFAULTS.get("roi", {})
    srt_cfg = DEFAULTS.get("srt", {})
    llm_cfg = DEFAULTS.get("llm", {})
    terms = tuple(part.strip() for part in _clean_text(watermark_text).split(",") if part.strip())
    return PipelineConfig(
        input_path=_source(path_text, uploaded), output_dir=_path(output_value),
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
        skip_unchanged_frames=DEFAULTS.get("ocr", {}).get("skip_unchanged_frames", True),
        captions_output=captions_output,
        captions=caption_config_from(DEFAULTS.get("captions"), srt_cfg.get("watermark_seconds", 15)),
        builder=BuilderConfig(
            similarity_threshold=srt_cfg.get("similarity_threshold", 85),
            watermark_seconds=srt_cfg.get("watermark_seconds", 15),
            minimum_duration_seconds=srt_cfg.get("minimum_duration_seconds", 0.15),
            bridge_gap_seconds=srt_cfg.get("bridge_gap_seconds", 0.12),
            minimum_samples=srt_cfg.get("minimum_samples", 2),
            watermark_terms=terms,
        ),
        llm=LLMConfig(
            enabled=llm_enabled, base_url=_clean_text(llm_base_url) or None,
            api_key=_clean_text(llm_api_key) or os.getenv("HARDSUB_LLM_API_KEY") or None,
            model=_clean_text(llm_model) or None,
            batch_size=llm_cfg.get("batch_size", 30),
            timeout_seconds=llm_cfg.get("timeout_seconds", 90),
        ),
        ffmpeg=_tool("ffmpeg"), ffprobe=_tool("ffprobe"),
    )


def _detect_preview(
    path_text: str | None, uploaded: str | None, output_text: str | None, auto_roi: bool,
    x_min: float, y_min: float, x_max: float, y_max: float,
) -> tuple[object, Path]:
    source = _source(path_text, uploaded)
    if not source.is_file():
        raise ValueError("Ảnh xem trước cần một tệp video, không phải thư mục.")
    output = _path(clean_path_text(output_text) or "output")
    manual = _roi(auto_roi, x_min, y_min, x_max, y_max)
    engine = OCREngine(
        prefer_gpu=DEFAULTS.get("ocr", {}).get("prefer_gpu", True),
        model_size=DEFAULTS.get("ocr", {}).get("model_size", "small"),
    ) if auto_roi else None
    roi_cfg = DEFAULTS.get("roi", {})
    raw_path = output / f"{source.stem}_roi_frame.jpg"
    result = detect_roi(
        source, engine, manual=manual,
        preview_path=output / f"{source.stem}_roi_preview.jpg",
        raw_preview_path=raw_path,
        sample_frames=roi_cfg.get("sample_frames", 30),
        max_sample_frames=roi_cfg.get("max_sample_frames", 90),
        probe_max_width=roi_cfg.get("probe_max_width", 960),
        minimum_support_frames=roi_cfg.get("minimum_support_frames", 2),
        padding_ratio=roi_cfg.get("padding_ratio", 0.03),
        ffmpeg=_tool("ffmpeg"), ffprobe=_tool("ffprobe"),
    )
    return result, raw_path


def preview_roi(
    path_text: str | None, uploaded: str | None, output_text: str | None, auto_roi: bool,
    x_min: float, y_min: float, x_max: float, y_max: float,
) -> tuple[str | None, str]:
    """Render one preview before committing to full video OCR."""
    try:
        result, _ = _detect_preview(path_text, uploaded, output_text, auto_roi,
                                    x_min, y_min, x_max, y_max)
        return _public_copy(result.preview_path), f"{result.message} ROI: {result.roi}"
    except Exception as exc:
        return None, f"Lỗi xem trước ROI: {exc}"


def preview_roi_editor(
    path_text: str | None, uploaded: str | None, output_text: str | None, auto_roi: bool,
    x_min: float, y_min: float, x_max: float, y_max: float,
) -> tuple[dict | object, str, float | object, float | object, float | object, float | object]:
    try:
        result, raw_path = _detect_preview(path_text, uploaded, output_text, auto_roi,
                                           x_min, y_min, x_max, y_max)
        src = "data:image/jpeg;base64," + base64.b64encode(raw_path.read_bytes()).decode("ascii")
        with Image.open(raw_path) as frame:
            width, height = frame.size
        display_width = round(min(width, 960, 700 * width / height))
        coords = [result.roi.x_min, result.roi.y_min, result.roi.x_max, result.roi.y_max]
        return {"src": src, "roi": coords, "display_width": display_width}, f"{result.message} Kéo khung xanh hoặc các góc để chỉnh ROI.", *coords
    except Exception as exc:
        return gr.skip(), f"Lỗi xem trước ROI: {exc}", *(gr.skip() for _ in range(4))


def roi_from_image(value: dict) -> tuple[bool, float, float, float, float, str]:
    roi = ROI(*(float(part) for part in value["roi"]))
    return False, roi.x_min, roi.y_min, roi.x_max, roi.y_max, "Đã chọn ROI thủ công trên ảnh."


def roi_from_sliders(value: dict, x_min: float, y_min: float, x_max: float, y_max: float):
    try:
        roi = ROI(x_min, y_min, x_max, y_max)
    except ValueError as exc:
        return gr.skip(), False, f"ROI chưa hợp lệ: {exc}"
    if not value or not value.get("src"):
        return gr.skip(), False, "Đã cập nhật tọa độ. Bấm Xem trước ROI để hiện ảnh."
    return {**value, "roi": [roi.x_min, roi.y_min, roi.x_max, roi.y_max]}, False, "Đã cập nhật khung ROI trên ảnh."


# The web queue runs one job at a time (default_concurrency_limit=1), so a single
# shared event is enough for the Stop button to reach the running job.
_STOP_LOCK = threading.Lock()
_ACTIVE_STOP: threading.Event | None = None


def request_stop() -> str:
    """Ask the running job to stop after the current frame and save its subtitles."""
    with _STOP_LOCK:
        event = _ACTIVE_STOP
    if event is None:
        return "Không có tác vụ nào đang chạy."
    event.set()
    return "Đang dừng... phụ đề đã quét sẽ được lưu vào thư mục đầu ra."


def run_ui(
    path_text: str | None, uploaded: str | None, output_text: str | None, language: str, fps: int,
    recursive: bool, preserve: bool, overwrite: bool, auto_roi: bool,
    x_min: float, y_min: float, x_max: float, y_max: float,
    watermark_text: str | None, llm_enabled: bool, llm_base_url: str | None,
    llm_api_key: str | None, llm_model: str | None, captions_output: str = "merged",
) -> Iterator[tuple[str, str, str | None, str | None]]:
    """Stream live events from the blocking pipeline into Gradio outputs."""
    try:
        config = _config(path_text, uploaded, output_text, language, fps, recursive,
                         preserve, overwrite, auto_roi, x_min, y_min, x_max, y_max,
                         watermark_text, llm_enabled, llm_base_url, llm_api_key, llm_model,
                         captions_output)
    except Exception as exc:
        yield f"Lỗi cấu hình: {exc}", str(exc), None, None
        return

    global _ACTIVE_STOP
    events: queue.Queue[tuple[str, object]] = queue.Queue()
    stop_event = threading.Event()
    with _STOP_LOCK:
        _ACTIVE_STOP = stop_event

    def worker() -> None:
        try:
            result = run_batch(config, callback=lambda event: events.put(("event", event)),
                               stop_event=stop_event)
            events.put(("summary", result))
        except Exception as exc:
            events.put(("error", exc))
        finally:
            events.put(("end", None))

    threading.Thread(target=worker, daemon=True).start()
    logs = [f"Đầu vào: {config.input_path}", f"Đầu ra: {config.output_dir}"]
    yield "Đang chuẩn bị...", "\n".join(logs), None, None
    try:
        yield from _stream_events(events, config, logs)
    finally:
        # Also reached when the browser tab is closed mid-run: stop the worker so
        # it saves the partial subtitles instead of running on unseen.
        stop_event.set()
        with _STOP_LOCK:
            if _ACTIVE_STOP is stop_event:
                _ACTIVE_STOP = None


def _stream_events(
    events: "queue.Queue[tuple[str, object]]", config: PipelineConfig, logs: list[str],
) -> Iterator[tuple[str, str, str | None, str | None]]:
    image_path: str | None = None
    source_preview: Path | None = None
    download_path: str | None = None
    status = "Đang chuẩn bị..."
    last_yield = 0.0
    while True:
        kind, value = events.get()
        if kind == "end":
            break
        frame_update = False
        if kind == "event":
            event = value
            assert isinstance(event, ProgressEvent)
            if event.phase == "roi":
                status = f"Đang tìm ROI: {event.video.name}"
                logs.append(status)
            elif event.phase == "ocr":
                if event.preview_path and event.preview_path.is_file() and event.preview_path != source_preview:
                    source_preview = event.preview_path
                    image_path = _public_copy(event.preview_path)
                if event.processed_frames:
                    frame_update = True
                    fraction = min(1.0, event.processed_frames / max(1, event.duration_seconds * config.fps))
                    detail = (f"Đang OCR: {event.video.name} — {event.processed_frames} khung hình, "
                              f"{event.processing_fps:.1f} FPS")
                    eta = (f"còn ~{format_clock(event.eta_seconds)}" if event.eta_seconds is not None
                           else "còn: đang tính...")
                    status = f"{_progress_bar(fraction)} {detail} · {eta}"
                    if event.processed_frames % 30 == 0:
                        logs.append(detail)
                else:
                    status = f"Đã chọn ROI: {event.roi}"
                    logs.append(f"{event.video.name}: {event.message} {event.roi}")
            elif event.phase == "done":
                status = f"Hoàn thành {event.video.name}: {event.message}"
                logs.append(status)
            elif event.phase == "failed":
                status = f"Lỗi {event.video.name}: {event.message}"
                logs.append(status)
            elif event.phase == "stopped":
                status = f"Đã dừng {event.video.name}: {event.message}"
                logs.append(status)
        elif kind == "summary":
            summary = value
            assert isinstance(summary, BatchSummary)
            for item in summary.files:
                logs.append(f"{item.status}: {item.video.name} → {item.output} ({item.message})")
            status = (f"Tổng kết: {summary.completed} hoàn thành, {summary.skipped} bỏ qua, "
                      f"{summary.failed} lỗi")
            stopped = next((item for item in summary.files if item.status == "stopped"), None)
            if stopped is not None:
                status += f" — đã dừng: {stopped.message}"
            if len(summary.files) == 1 and summary.files[0].output.is_file():
                download_path = _public_copy(summary.files[0].output)
        else:
            status = f"Lỗi xử lý: {value}"
            logs.append(status)
        # Frame-by-frame OCR events arrive many times per second; refreshing the
        # page that often makes the output area flicker, so throttle those.
        now = time.monotonic()
        if frame_update and now - last_yield < UI_REFRESH_SECONDS:
            continue
        last_yield = now
        yield status, "\n".join(logs[-60:]), image_path, download_path


def _initial_dir(current: str | None) -> str | None:
    """Start the dialog next to the path already typed, when it exists."""
    text = clean_path_text(current)
    if not text:
        return None
    path = Path(text).expanduser()
    for candidate in (path, path.parent):
        if candidate.is_dir():
            return str(candidate)
    return None


def _picker(kind: str):  # type: ignore[no-untyped-def]
    def pick(current: str | None) -> str:
        try:
            chosen = pick_local_path(kind, _initial_dir(current))
        except Exception as exc:  # e.g. no desktop session available
            raise gr.Error(str(exc)) from exc
        return chosen if chosen else (current or "")  # Cancel keeps what was there
    return pick


def build_app(local_mode: bool = False) -> gr.Blocks:
    """``local_mode``: browser and server are the same computer, so files are
    chosen with the native dialog (path only, nothing uploaded). On a remote
    server the dialog would open on the server, so the upload box is used."""
    WEB_CACHE.mkdir(parents=True, exist_ok=True)
    default_output = os.getenv("HARDSUB_OUTPUT_DIR", str(ROOT / "output"))
    default_input = os.getenv("HARDSUB_INPUT_DIR", "")
    with gr.Blocks(title="Hardsub2srt") as demo:
        gr.Markdown("# Hardsub2srt\nTrích phụ đề cứng từ video thành tệp SRT.")
        with gr.Row(equal_height=True):
            input_path = gr.Textbox(
                value=default_input, scale=4,
                label="Đường dẫn video hoặc thư mục" + ("" if local_mode else " trên máy chủ"),
                placeholder=r"Ví dụ: C:\Videos\phim.mp4 — dán cả đường dẫn có dấu ngoặc kép cũng được",
            )
            with gr.Column(scale=1, min_width=160, visible=local_mode):
                pick_file = gr.Button("Chọn tệp video…", size="sm")
                pick_folder = gr.Button("Chọn thư mục…", size="sm")
            upload = gr.File(label="Hoặc tải lên một video", file_types=VIDEO_TYPES, type="filepath",
                             visible=not local_mode, scale=3)
        with gr.Row(equal_height=False):
            output_dir = gr.Textbox(value=default_output, label="Thư mục đầu ra", scale=4)
            pick_output = gr.Button("Chọn thư mục…", size="sm", scale=1, min_width=160, visible=local_mode)
        # Clean pasted paths when leaving the box or pressing Enter (not on every
        # keystroke, so a space typed at the end of "My Videos" is not removed).
        for box in (input_path, output_dir):
            box.blur(clean_path_text, inputs=[box], outputs=[box], queue=False, show_progress="hidden")
            box.submit(clean_path_text, inputs=[box], outputs=[box], queue=False, show_progress="hidden")
        # queue=False: the dialog must open even while an OCR job holds the queue.
        pick_file.click(_picker("file"), inputs=[input_path], outputs=[input_path], queue=False)
        pick_folder.click(_picker("folder"), inputs=[input_path], outputs=[input_path], queue=False)
        pick_output.click(_picker("folder"), inputs=[output_dir], outputs=[output_dir], queue=False)
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
            gr.Markdown("Bấm xem trước, rồi kéo khung xanh hoặc các góc ngay trên ảnh. Sửa tọa độ cũng cập nhật khung; cả hai cách đều chuyển sang ROI thủ công.")
            with gr.Row():
                x_min = gr.Slider(0, 1, value=0, step=0.01, label="X nhỏ nhất")
                y_min = gr.Slider(0, 1, value=0.65, step=0.01, label="Y nhỏ nhất")
                x_max = gr.Slider(0, 1, value=1, step=0.01, label="X lớn nhất")
                y_max = gr.Slider(0, 1, value=1, step=0.01, label="Y lớn nhất")
            preview_button = gr.Button("Xem trước ROI")
            roi_editor = gr.HTML(
                value={"src": "", "roi": [0, 0.65, 1, 1], "display_width": 0},
                html_template=HTML_TEMPLATE, css_template=CSS_TEMPLATE,
                js_on_load=JS_ON_LOAD, label="Ảnh xem trước ROI",
            )
            with gr.Accordion("Ảnh ROI khi xử lý", open=False):
                preview_image = gr.Image(label="ROI đã dùng", interactive=False, height=420)
        watermark_terms = gr.Textbox(value=", ".join(DEFAULTS.get("srt", {}).get("watermark_terms", [])),
                                    label="Từ khóa watermark cần bỏ qua (ngăn cách bằng dấu phẩy)",
                                    placeholder="Ví dụ: tên logo")
        captions_output = gr.Dropdown(
            choices=[("Gộp vào tệp SRT (hiện ở đầu màn hình)", "merged"),
                     ("Lưu tệp riêng <tên>.captions.srt", "separate"), ("Bỏ qua", "off")],
            value=DEFAULTS.get("captions", {}).get("output", "merged"),
            label="Câu chú thích/biên tập ngoài vùng hội thoại (tự loại logo, quảng cáo, biển hiệu)",
        )
        with gr.Accordion("Sửa lỗi OCR bằng LLM (tùy chọn)", open=False):
            llm_enabled = gr.Checkbox(value=False, label="Bật sửa OCR bằng LLM")
            llm_base_url = gr.Textbox(label="Base URL", placeholder="http://localhost:11434/v1")
            llm_api_key = gr.Textbox(label="API Key", type="password")
            llm_model = gr.Textbox(label="Tên mô hình")
        with gr.Row():
            run_button = gr.Button("Bắt đầu xử lý", variant="primary", scale=4)
            stop_button = gr.Button("Dừng và lưu", variant="stop", scale=1)
        status = gr.Textbox(label="Trạng thái", interactive=False)
        logs = gr.Textbox(label="Nhật ký trực tiếp", lines=12, interactive=False)
        download = gr.File(label="Tải tệp SRT (chạy một video)", interactive=False)

        preview_button.click(
            preview_roi_editor,
            inputs=[input_path, upload, output_dir, auto_roi, x_min, y_min, x_max, y_max],
            outputs=[roi_editor, status, x_min, y_min, x_max, y_max],
        )
        roi_editor.submit(
            roi_from_image, inputs=[roi_editor],
            outputs=[auto_roi, x_min, y_min, x_max, y_max, status],
        )
        for slider in (x_min, y_min, x_max, y_max):
            slider.input(
                roi_from_sliders,
                inputs=[roi_editor, x_min, y_min, x_max, y_max],
                outputs=[roi_editor, auto_roi, status], queue=False,
            )
        run_button.click(
            run_ui,
            inputs=[input_path, upload, output_dir, language, fps, recursive, preserve,
                    overwrite, auto_roi, x_min, y_min, x_max, y_max, watermark_terms,
                    llm_enabled, llm_base_url, llm_api_key, llm_model, captions_output],
            outputs=[status, logs, preview_image, download],
            show_progress="hidden",
        )
        # queue=False: the job queue is busy with the running job, so the Stop
        # click must bypass it to reach that job.
        stop_button.click(request_stop, outputs=[status], queue=False, show_progress="hidden")
    return demo


if __name__ == "__main__":
    user, password = os.getenv("HARDSUB_WEB_USER"), os.getenv("HARDSUB_WEB_PASSWORD")
    if bool(user) != bool(password):
        raise SystemExit("Phải đặt cả HARDSUB_WEB_USER và HARDSUB_WEB_PASSWORD.")
    auth = (user, password) if user and password else None
    local_windows = os.name == "nt"
    server_name = os.getenv("HARDSUB_WEB_HOST") or ("127.0.0.1" if local_windows else "0.0.0.0")
    server_port = int(os.getenv("HARDSUB_WEB_PORT", "7860"))
    local_mode = server_name in {"127.0.0.1", "localhost", "::1"}
    build_app(local_mode=local_mode).queue(default_concurrency_limit=1).launch(
        server_name=server_name,
        server_port=server_port, inbrowser=local_windows, auth=auth,
        allowed_paths=[str(WEB_CACHE)], show_error=True,
    )
