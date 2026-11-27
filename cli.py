"""Headless CLI for sequential hard-subtitle extraction."""

from __future__ import annotations

import argparse
import signal
import threading
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from core.llm_refiner import LLMConfig
from core.captions import caption_config_from, screen_text_config_from
from core.paths import clean_path_text
from core.pipeline import PipelineConfig, ProgressEvent, format_clock, run_batch
from core.roi_detector import ROI
from core.srt_builder import BuilderConfig


def _setting(cli_value, config_value):
    return config_value if cli_value is None else cli_value


def _clean_path(value: str) -> Path:
    r"""Accept pasted paths such as "C:\Videos\a.mp4" with stray quotes."""
    return Path(clean_path_text(value))


def _executable(value: str, name: str) -> str:
    if shutil.which(value) or Path(value).is_file():
        return value
    project_tools = Path(__file__).parent / ".tools" / "ffmpeg"
    if value == name and project_tools.is_dir():
        match = next(project_tools.rglob(name + ".exe"), None)
        if match:
            return str(match)
    return value


def _load_settings(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config.yaml phải là YAML object")
    return data


def _options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trích phụ đề cứng từ video thành SRT")
    parser.add_argument("--input", required=True, type=_clean_path, help="Tệp video hoặc thư mục đầu vào")
    parser.add_argument("--output", type=_clean_path, help="Thư mục lưu SRT")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--lang", choices=("auto", "ch", "vi", "ja", "en"))
    parser.add_argument("--fps", type=float)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--preserve-subfolders", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--auto-roi", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--roi", nargs=4, type=float, metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"))
    parser.add_argument("--watermark-term", action="append", dest="watermark_terms",
                        help="Loại ô chữ chứa từ khóa watermark (có thể dùng nhiều lần)")
    parser.add_argument("--llm-refine", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--llm-base-url")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-model")
    parser.add_argument("--captions", choices=("merged", "separate", "off"),
                        help="Chú thích và tin nhắn trên màn hình: gộp vào SRT, tệp riêng, hoặc bỏ qua")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    return parser.parse_args()


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    console = Console()
    args = _options()
    try:
        config_path = args.config.expanduser().resolve()
        settings = _load_settings(config_path)
        input_settings = settings.get("input", {})
        output_settings = settings.get("output", {})
        video_settings = settings.get("video", {})
        roi_settings = settings.get("roi", {})
        ocr_settings = settings.get("ocr", {})
        srt_settings = settings.get("srt", {})
        llm_settings = settings.get("llm", {})
        caption_settings = settings.get("captions", {})
        output = args.output or Path(output_settings.get("directory", "output"))
        if not output.is_absolute():
            output = config_path.parent / output
        manual = args.roi if args.roi is not None else roi_settings.get("manual")
        manual_roi = ROI(*manual) if manual is not None else None
        builder = BuilderConfig(
            similarity_threshold=srt_settings.get("similarity_threshold", 85),
            watermark_seconds=srt_settings.get("watermark_seconds", 15),
            minimum_duration_seconds=srt_settings.get("minimum_duration_seconds", 0.15),
            bridge_gap_seconds=srt_settings.get("bridge_gap_seconds", 0.12),
            minimum_samples=srt_settings.get("minimum_samples", 2),
            watermark_terms=tuple(_setting(args.watermark_terms, srt_settings.get("watermark_terms", []))),
        )
        llm = LLMConfig(
            enabled=_setting(args.llm_refine, llm_settings.get("enabled", False)),
            base_url=args.llm_base_url or llm_settings.get("base_url"),
            api_key=args.llm_api_key or os.getenv("HARDSUB_LLM_API_KEY") or llm_settings.get("api_key"),
            model=args.llm_model or llm_settings.get("model"),
            batch_size=llm_settings.get("batch_size", 30),
            timeout_seconds=llm_settings.get("timeout_seconds", 90),
        )
        config = PipelineConfig(
            input_path=args.input,
            output_dir=output,
            language=_setting(args.lang, ocr_settings.get("language", "auto")),
            fps=_setting(args.fps, video_settings.get("sample_fps", 3)),
            recursive=_setting(args.recursive, input_settings.get("recursive", False)),
            preserve_subfolders=_setting(args.preserve_subfolders, output_settings.get("preserve_subfolders", False)),
            overwrite=_setting(args.overwrite, output_settings.get("overwrite", False)),
            auto_roi=_setting(args.auto_roi, roi_settings.get("mode", "auto") == "auto"),
            manual_roi=manual_roi,
            roi_sample_frames=roi_settings.get("sample_frames", 30),
            roi_max_sample_frames=roi_settings.get("max_sample_frames", 90),
            roi_probe_max_width=roi_settings.get("probe_max_width", 960),
            roi_minimum_support_frames=roi_settings.get("minimum_support_frames", 2),
            roi_padding_ratio=roi_settings.get("padding_ratio", 0.03),
            prefer_gpu=ocr_settings.get("prefer_gpu", True),
            ocr_model_size=ocr_settings.get("model_size", "small"),
            skip_unchanged_frames=ocr_settings.get("skip_unchanged_frames", True),
            captions_output=_setting(args.captions, caption_settings.get("output", "merged")),
            captions=caption_config_from(caption_settings, srt_settings.get("watermark_seconds", 15)),
            screen_text=screen_text_config_from(caption_settings.get("screen_text")),
            builder=builder,
            llm=llm,
            ffmpeg=_executable(args.ffmpeg or video_settings.get("ffmpeg", "ffmpeg"), "ffmpeg"),
            ffprobe=_executable(args.ffprobe or video_settings.get("ffprobe", "ffprobe"), "ffprobe"),
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        console.print(f"[red]Cấu hình lỗi:[/red] {exc}")
        return 2

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    tasks: dict[Path, int] = {}
    progress = Progress(TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(),
                        TextColumn("{task.fields[rate]}"), TimeElapsedColumn(),
                        TextColumn("{task.fields[eta]}"), console=console)

    def update(event: ProgressEvent) -> None:
        if event.phase == "roi":
            tasks[event.video] = progress.add_task(event.video.name, total=max(event.duration_seconds, 1e-6),
                                                   rate="đang tìm ROI", eta="")
        elif event.phase == "ocr":
            if event.roi is not None and event.processed_frames == 0:
                console.print(f"ROI {event.video.name}: {event.roi}")
            task = tasks.get(event.video)
            if task is not None:
                if event.eta_seconds is not None:
                    eta = f"còn ~{format_clock(event.eta_seconds)}"
                else:
                    eta = "còn: đang tính..." if event.processed_frames else ""
                progress.update(task, completed=min(event.duration_seconds, event.processed_frames / config.fps),
                                rate=f"{event.processing_fps:.1f} FPS" if event.processed_frames else "đang OCR",
                                eta=eta)
        elif event.phase == "done":
            task = tasks.get(event.video)
            if task is not None:
                progress.update(task, completed=event.duration_seconds,
                                rate=f"{event.processing_fps:.1f} FPS", eta="xong")
        elif event.phase == "failed":
            console.print(f"[red]Lỗi {event.video.name}:[/red] {event.message}")
        elif event.phase == "stopped":
            if (task := tasks.get(event.video)) is not None:
                progress.update(task, eta="đã dừng")
            console.print(f"[yellow]Đã dừng {event.video.name}:[/yellow] {event.message}")

    # First Ctrl+C: stop after the current frame and save what was read.
    # Second Ctrl+C: quit immediately without saving.
    stop_event = threading.Event()

    def request_stop(signum, frame) -> None:  # type: ignore[no-untyped-def]
        if stop_event.is_set():
            raise KeyboardInterrupt
        stop_event.set()
        console.print("[yellow]Đang dừng... phụ đề đã quét sẽ được lưu. "
                      "Nhấn Ctrl+C lần nữa để thoát ngay (không lưu).[/yellow]")

    previous_handler = signal.signal(signal.SIGINT, request_stop)
    console.print("Nhấn Ctrl+C để dừng giữa chừng và lưu phụ đề đã quét.")
    try:
        with progress:
            summary = run_batch(config, callback=update, stop_event=stop_event)
    except KeyboardInterrupt:
        console.print("[red]Đã thoát ngay; không lưu phụ đề.[/red]")
        return 130
    except (OSError, ValueError) as exc:
        console.print(f"[red]Không chạy được:[/red] {exc}")
        return 2
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    for result in summary.files:
        paths = result.outputs or (result.output,)
        console.print(f"{result.status}: {result.video.name} → "
                      f"{', '.join(str(path) for path in paths)} "
                      f"({result.subtitle_count} phụ đề) {result.message}")
    console.print(f"Tổng kết: {summary.completed} hoàn thành, {summary.skipped} bỏ qua, "
                  f"{summary.failed} lỗi" + (" — đã dừng giữa chừng" if summary.stopped else ""))
    if summary.stopped:
        return 130
    if not summary.files:
        console.print("Không tìm thấy video phù hợp.")
        return 1
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
