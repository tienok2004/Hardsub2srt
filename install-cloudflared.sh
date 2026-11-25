#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "$(uname -m)" in
  x86_64|amd64) asset=cloudflared-linux-amd64 ;;
  aarch64|arm64) asset=cloudflared-linux-arm64 ;;
  *) echo "Kiến trúc CPU chưa hỗ trợ: $(uname -m)" >&2; exit 1 ;;
esac

command -v curl >/dev/null || { echo "Cần cài curl trước." >&2; exit 1; }
mkdir -p "$root_dir/.tools"
temporary="$(mktemp "$root_dir/.tools/cloudflared.XXXXXXXX")"
trap 'rm -f -- "$temporary"' EXIT
curl --fail --location --show-error --silent \
  "https://github.com/cloudflare/cloudflared/releases/latest/download/$asset" \
  --output "$temporary"
chmod 755 "$temporary"
"$temporary" --version
mv -f -- "$temporary" "$root_dir/.tools/cloudflared"
echo "Đã cài cloudflared tại $root_dir/.tools/cloudflared"
