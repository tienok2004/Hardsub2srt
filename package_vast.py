"""Create a small source archive for a rented Vast.ai instance."""

from __future__ import annotations

import tarfile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "dist" / "hardsub2srt-vast.tar.gz"
FILES = [
    "app.py",
    "cli.py",
    "config.yaml",
    "requirements.txt",
    "requirements-gpu.txt",
    "setup-vast.sh",
    "run-vast.sh",
    "install-cloudflared.sh",
    "README.md",
]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sources = [ROOT / name for name in FILES]
    sources += sorted((ROOT / "core").glob("*.py"))
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise SystemExit("Thiếu tệp nguồn: " + ", ".join(missing))

    OUTPUT.parent.mkdir(exist_ok=True)
    with tarfile.open(OUTPUT, "w:gz") as archive:
        for path in sources:
            archive.add(path, arcname="hardsub2srt/" + path.relative_to(ROOT).as_posix())
    print(f"Đã tạo {OUTPUT} ({OUTPUT.stat().st_size:,} byte)")


if __name__ == "__main__":
    main()
