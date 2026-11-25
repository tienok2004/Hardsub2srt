#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"

if [[ -z "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
  echo "Thiếu CLOUDFLARE_TUNNEL_TOKEN. Hãy đặt biến môi trường trước khi chạy." >&2
  exit 1
fi
if [[ -z "${HARDSUB_WEB_USER:-}" || -z "${HARDSUB_WEB_PASSWORD:-}" ]]; then
  echo "Cần đặt HARDSUB_WEB_USER và HARDSUB_WEB_PASSWORD để bảo vệ giao diện công khai." >&2
  exit 1
fi

python_bin="$root_dir/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "Thiếu .venv/bin/python. Hãy cài requirements-gpu.txt theo README trước." >&2
  exit 1
fi

if command -v cloudflared >/dev/null 2>&1; then
  cloudflared_bin="$(command -v cloudflared)"
elif [[ -x "$root_dir/.tools/cloudflared" ]]; then
  cloudflared_bin="$root_dir/.tools/cloudflared"
else
  echo "Thiếu cloudflared. Chạy bash install-cloudflared.sh trước." >&2
  exit 1
fi

export HARDSUB_WEB_HOST=127.0.0.1
export HARDSUB_INPUT_DIR="${HARDSUB_INPUT_DIR:-$root_dir/input}"
export HARDSUB_OUTPUT_DIR="${HARDSUB_OUTPUT_DIR:-$root_dir/output}"
mkdir -p "$HARDSUB_INPUT_DIR" "$HARDSUB_OUTPUT_DIR"
tunnel_token="$CLOUDFLARE_TUNNEL_TOKEN"
unset CLOUDFLARE_TUNNEL_TOKEN

app_pid=''
tunnel_pid=''
cleanup() {
  [[ -z "$tunnel_pid" ]] || kill "$tunnel_pid" 2>/dev/null || true
  [[ -z "$app_pid" ]] || kill "$app_pid" 2>/dev/null || true
  [[ -z "$tunnel_pid" ]] || wait "$tunnel_pid" 2>/dev/null || true
  [[ -z "$app_pid" ]] || wait "$app_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$python_bin" app.py &
app_pid=$!

for ((attempt = 0; attempt < 120; attempt++)); do
  if ! kill -0 "$app_pid" 2>/dev/null; then
    echo "Web UI đã dừng trước khi sẵn sàng." >&2
    exit 1
  fi
  if "$python_bin" -c 'import socket; s=socket.socket(); s.settimeout(0.5); s.connect(("127.0.0.1",7860)); s.close()' 2>/dev/null; then
    break
  fi
  sleep 1
done

if ((attempt == 120)); then
  echo "Web UI không mở cổng 7860 sau 120 giây." >&2
  exit 1
fi

echo "Web UI đã sẵn sàng trên 127.0.0.1:7860; đang kết nối Cloudflare Tunnel."
TUNNEL_TOKEN="$tunnel_token" "$cloudflared_bin" tunnel --no-autoupdate run &
tunnel_pid=$!
status=0
wait -n "$app_pid" "$tunnel_pid" || status=$?
echo "Web UI hoặc Cloudflare Tunnel đã dừng (mã $status)." >&2
exit "$status"
