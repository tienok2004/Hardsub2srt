"""Inspect NVIDIA/ONNX Runtime availability and choose an execution provider.

An available CUDA provider is only a preliminary signal. The OCR stage must call
``create_ort_session`` with its real model to verify session initialization.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuStatus:
    device_name: str | None
    onnxruntime_installed: bool
    cuda_provider_available: bool
    cuda_session_initialized: bool
    active_provider: str


def _device_name(nvidia_smi: str = "nvidia-smi") -> str | None:
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return next((name.strip() for name in result.stdout.splitlines() if name.strip()), None)


def _onnxruntime() -> Any | None:
    try:
        return importlib.import_module("onnxruntime")
    except ImportError:
        return None


_DLL_DIRS_ADDED = False


def _add_nvidia_dll_dirs() -> None:
    """Expose pip-installed CUDA/cuDNN ``bin`` folders to the Windows DLL loader.

    ``preload_dlls`` loads ``cudnn64_9.dll`` itself, but cuDNN 9 opens its
    sub-libraries (``cudnn_engines_*``, ``cudnn_cnn64_9`` ...) later by name. Without
    their folder on the search path the first convolution fails with
    ``CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`` and ONNX Runtime drops to CPU.
    """
    global _DLL_DIRS_ADDED
    if _DLL_DIRS_ADDED or os.name != "nt":
        return
    _DLL_DIRS_ADDED = True
    for site_dir in map(Path, sys.path):
        nvidia = site_dir / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in sorted(nvidia.glob("*/bin")):
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            try:
                os.add_dll_directory(str(bin_dir))
            except OSError:
                pass
        return


def _dummy_inputs(session: Any) -> dict[str, Any]:
    import numpy as np

    feeds = {}
    for item in session.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else (1 if index == 0 else 64)
                 for index, dim in enumerate(item.shape)]
        feeds[item.name] = np.zeros(shape, dtype=np.float32)
    return feeds


def _runs_on_cuda(ort: Any, model_path: str | Path) -> bool:
    """Run one inference on CUDA only; session creation alone can succeed while
    the first convolution still fails inside cuDNN."""
    session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
    if "CUDAExecutionProvider" not in session.get_providers():
        return False
    session.run(None, _dummy_inputs(session))
    return True


def _preload_cuda_libraries(ort: Any) -> None:
    """Load CUDA/cuDNN from the venv before ONNX Runtime opens its CUDA provider."""
    _add_nvidia_dll_dirs()
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        try:
            preload()
        except (OSError, RuntimeError) as exc:
            LOGGER.warning("Không nạp trước được thư viện CUDA/cuDNN: %s", exc)


def inspect_gpu(model_path: str | Path | None = None, *, nvidia_smi: str = "nvidia-smi") -> GpuStatus:
    """Check device/provider availability; optionally prove CUDA session startup.

    Without a model, ``cuda_session_initialized`` is false even when the provider
    is listed, since listing a provider does not prove that CUDA can load a model.
    """
    name = _device_name(nvidia_smi)
    ort = _onnxruntime()
    available = ort is not None and "CUDAExecutionProvider" in ort.get_available_providers()
    initialized = False
    if available and model_path is not None:
        try:
            _preload_cuda_libraries(ort)
            initialized = _runs_on_cuda(ort, model_path)
        except Exception as exc:  # CUDA DLL/model failures vary across ORT versions.
            LOGGER.warning("Không khởi tạo được phiên ONNX Runtime CUDA: %s", exc)

    active = "CUDAExecutionProvider" if initialized else "CPUExecutionProvider"
    if initialized:
        LOGGER.info("GPU đang dùng: %s (%s)", name or "không xác định", active)
    elif available and model_path is None:
        LOGGER.info("Có CUDAExecutionProvider; cần mô hình OCR để xác minh khởi tạo phiên GPU.")
    else:
        LOGGER.warning("Không có phiên CUDA hoạt động; sẽ dùng CPU khi chạy OCR.")
    return GpuStatus(name, ort is not None, available, initialized, active)


def create_ort_session(model_path: str | Path, *, prefer_gpu: bool = True) -> Any:
    """Open a real model on CUDA when possible, with a clear CPU fallback."""
    ort = _onnxruntime()
    if ort is None:
        raise RuntimeError("Cần cài onnxruntime hoặc onnxruntime-gpu để chạy OCR.")
    if prefer_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            _preload_cuda_libraries(ort)
            session = ort.InferenceSession(
                str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            if "CUDAExecutionProvider" in session.get_providers():
                LOGGER.info("Đã khởi tạo CUDAExecutionProvider trên %s", _device_name() or "GPU không rõ tên")
                return session
        except Exception as exc:
            LOGGER.warning("Không khởi tạo được CUDAExecutionProvider: %s", exc)
    LOGGER.warning("Chuyển sang CPUExecutionProvider.")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def main() -> int:
    # Some Windows shells expose a non-UTF-8 pipe despite UTF-8 source files.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Kiểm tra khả năng chạy OCR bằng NVIDIA GPU")
    parser.add_argument("--model", type=Path, help="Mô hình ONNX để thử khởi tạo CUDA thực tế")
    parser.add_argument("--require-cuda", action="store_true", help="Trả mã lỗi nếu không khởi tạo được phiên CUDA")
    args = parser.parse_args()
    if args.require_cuda and args.model is None:
        parser.error("--require-cuda cần --model")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    status = inspect_gpu(args.model)
    print(f"Thiết bị NVIDIA: {status.device_name or 'không phát hiện'}")
    print(f"ONNX Runtime: {'đã cài' if status.onnxruntime_installed else 'chưa cài'}")
    print(f"CUDA provider: {'có' if status.cuda_provider_available else 'không'}")
    print(f"Phiên CUDA: {'đã xác minh' if status.cuda_session_initialized else 'chưa xác minh'}")
    print(f"Provider được chọn: {status.active_provider}")
    return 0 if not args.require_cuda or status.cuda_session_initialized else 1


if __name__ == "__main__":
    raise SystemExit(main())

