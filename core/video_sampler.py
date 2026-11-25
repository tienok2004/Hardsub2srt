"""Read video metadata and stream uniformly sampled RGB frames through FFmpeg."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator


class VideoError(RuntimeError):
    """Raised when a video cannot be inspected or decoded."""


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    duration_seconds: float
    source_fps: float
    frame_count: int | None


@dataclass(frozen=True)
class SampledFrame:
    timestamp_seconds: float
    width: int
    height: int
    rgb24: bytes


def _fraction(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: str | Path, *, ffprobe: str = "ffprobe") -> VideoMetadata:
    """Return metadata for the first video stream, or raise ``VideoError``."""
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise VideoError(f"Không tìm thấy tệp video: {video_path}")
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
        "-of", "json", str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        duration = float(data.get("format", {}).get("duration") or stream.get("duration") or 0)
        fps = _fraction(stream.get("avg_frame_rate")) or _fraction(stream.get("r_frame_rate"))
        frame_count = int(stream["nb_frames"]) if stream.get("nb_frames") else None
        if width <= 0 or height <= 0 or duration <= 0 or not math.isfinite(duration):
            raise ValueError("Kích thước hoặc thời lượng video không hợp lệ")
        return VideoMetadata(video_path, width, height, duration, fps, frame_count)
    except FileNotFoundError as exc:
        raise VideoError(f"Không tìm thấy ffprobe: {ffprobe}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoError(f"Hết thời gian đọc thông tin video: {video_path}") from exc
    except subprocess.CalledProcessError as exc:
        raise VideoError(f"ffprobe không đọc được {video_path}: {exc.stderr.strip()}") from exc
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VideoError(f"Thông tin video không hợp lệ: {video_path}: {exc}") from exc


def _read_frame(stream: object, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)  # type: ignore[attr-defined]
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def sample_frames(
    path: str | Path,
    *,
    fps: float = 3.0,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    metadata: VideoMetadata | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> Iterator[SampledFrame]:
    """Yield RGB24 frames at 0, 1/fps, 2/fps ... seconds.

    Frames are decoded in one FFmpeg process. ``-noautorotate`` keeps raw frame
    dimensions in sync with ffprobe; display rotation can be handled in the OCR
    stage if needed. The caller may close this generator to stop FFmpeg early.
    """
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps phải là số dương hữu hạn")
    info = metadata or probe_video(path, ffprobe=ffprobe)
    width, height = info.width, info.height
    filter_graph = f"fps={fps:.12g}"
    if crop is not None:
        x1, y1, x2, y2 = crop
        if not (0 <= x1 < x2 <= info.width and 0 <= y1 < y2 <= info.height):
            raise ValueError("Vùng cắt nằm ngoài khung hình")
        width, height = x2 - x1, y2 - y1
        filter_graph += f",crop={width}:{height}:{x1}:{y1}:exact=1"
    frame_bytes = width * height * 3
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-noautorotate",
        "-i", str(info.path), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", filter_graph, "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise VideoError(f"Không tìm thấy ffmpeg: {ffmpeg}") from exc
    index = 0
    try:
        assert process.stdout is not None
        while True:
            payload = _read_frame(process.stdout, frame_bytes)
            if not payload:
                break
            if len(payload) != frame_bytes:
                raise VideoError(f"Khung hình cuối bị thiếu dữ liệu: {info.path}")
            yield SampledFrame(index / fps, width, height, payload)
            index += 1
        return_code = process.wait(timeout=30)
        if return_code != 0:
            assert process.stderr is not None
            detail = process.stderr.read().decode("utf-8", errors="replace").strip()
            raise VideoError(f"ffmpeg không giải mã được {info.path}: {detail}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()


def sample_frame_at(
    path: str | Path,
    timestamp_seconds: float,
    *,
    max_width: int = 960,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    metadata: VideoMetadata | None = None,
) -> SampledFrame:
    """Seek directly to one time point for sparse ROI probing.

    Unlike ``sample_frames``, this does not decode everything before the chosen
    time. The smaller RGB frame bounds both FFmpeg transfer and text detection.
    """
    info = metadata or probe_video(path, ffprobe=ffprobe)
    if not math.isfinite(timestamp_seconds) or not 0 <= timestamp_seconds < info.duration_seconds:
        raise ValueError("Mốc thời gian phải nằm trong video")
    if max_width <= 0:
        raise ValueError("max_width phải là số dương")
    scale = min(1.0, max_width / info.width)
    width = max(1, round(info.width * scale))
    height = max(1, round(info.height * scale))
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-noautorotate",
        "-ss", f"{timestamp_seconds:.6f}", "-i", str(info.path),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", "1",
        "-vf", f"scale={width}:{height}", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=40)
    except FileNotFoundError as exc:
        raise VideoError(f"Không tìm thấy ffmpeg: {ffmpeg}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VideoError(f"Hết thời gian lấy khung hình tại {timestamp_seconds:.3f}s") from exc
    if result.returncode or len(result.stdout) != width * height * 3:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise VideoError(f"Không lấy được khung hình tại {timestamp_seconds:.3f}s: {detail}")
    return SampledFrame(timestamp_seconds, width, height, result.stdout)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Đọc metadata và lấy mẫu khung hình video")
    parser.add_argument("video", type=Path)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    try:
        info = probe_video(args.video, ffprobe=args.ffprobe)
        print(f"Video: {info.path}")
        print(f"Kích thước: {info.width}x{info.height}")
        print(f"Thời lượng: {info.duration_seconds:.3f}s; FPS nguồn: {info.source_fps:.3f}")
        count = sum(1 for _ in sample_frames(info.path, fps=args.fps, ffmpeg=args.ffmpeg, metadata=info))
        print(f"Đã lấy {count} khung hình tại {args.fps:g} FPS")
    except (VideoError, ValueError) as exc:
        parser.exit(1, f"Lỗi: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

