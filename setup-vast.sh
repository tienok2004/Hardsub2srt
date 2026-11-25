#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

if ! command -v apt-get >/dev/null 2>&1; then
  echo "Script này cần image Vast.ai dùng Ubuntu hoặc Debian (apt-get)." >&2
  exit 1
fi

if (( EUID == 0 )); then
  apt=(apt-get)
elif command -v sudo >/dev/null 2>&1; then
  apt=(sudo apt-get)
else
  echo "Cần quyền root hoặc sudo để cài FFmpeg và thư viện hệ thống." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
"${apt[@]}" update
"${apt[@]}" install -y --no-install-recommends \
  python3 python3-venv python3-pip ffmpeg libgl1 libgomp1 libglib2.0-0 \
  zlib1g ca-certificates curl tmux

python3 -c 'import sys; print("Python:", sys.version.split()[0]); sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "Cần Python 3.11 trở lên cho ONNX Runtime GPU 1.26. Hãy chọn image Vast.ai mới hơn." >&2
  exit 1
}

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
PIP_NO_CACHE_DIR=1 .venv/bin/python -m pip install -r requirements-gpu.txt
bash install-cloudflared.sh

mkdir -p input output
ffmpeg -version | sed -n '1p'
.tools/cloudflared --version

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name --format=csv,noheader || true
else
  echo "Cảnh báo: chưa thấy nvidia-smi trong instance." >&2
fi

model_path="$(.venv/bin/python -c 'from pathlib import Path; import rapidocr; print(Path(rapidocr.__file__).parent / "models" / "PP-OCRv6_det_small.onnx")')"
if [[ -f "$model_path" ]]; then
  if ! .venv/bin/python -m core.gpu --model "$model_path" --require-cuda; then
    echo "Cảnh báo: OCR vẫn dùng CPU. Xem lỗi CUDA ở trên và kiểm tra driver NVIDIA của instance." >&2
  fi
else
  echo "Chưa có mô hình OCR mặc định để thử phiên CUDA; ứng dụng sẽ tải khi chạy lần đầu." >&2
fi

echo "Đã cài xong. Hãy đặt thông tin đăng nhập và tunnel token, rồi chạy bash run-vast.sh."
