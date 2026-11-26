"""Clean file paths pasted by users and pick local files without uploading."""

from __future__ import annotations

import os
import subprocess
import sys
from urllib.parse import unquote, urlparse

# Wrapping characters that "Copy as path", chat apps and documents add around paths.
_WRAPPERS = "\"'`“”‘’«»‹›「」『』"
# Never valid in a Windows file name; on Linux they are legal but almost always
# come from copy/paste noise, so they are only stripped on Windows.
_WINDOWS_INVALID = '<>"|?*'
_INVISIBLE = {"\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\u2066", "\u2067", "\u2068",
              "\u2069", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\ufeff"}


def clean_path_text(value: str | None, *, windows: bool | None = None) -> str:
    """Turn a pasted path into a plain one.

    Handles ``"C:\\Videos\\a.mp4"`` from Explorer's "Copy as path", ``file:///``
    links, the ``\\\\?\\`` long-path prefix, invisible direction marks, and stray
    characters such as ``?`` or ``*`` that can never be part of a Windows path.
    """
    if value is None:
        return ""
    windows = os.name == "nt" if windows is None else windows
    text = "".join(ch for ch in value if ch not in _INVISIBLE and (ch >= " " or ch in "\t\n"))
    # A multi-line paste: keep the first line that has content.
    text = next((line for line in text.splitlines() if line.strip()), "").replace("\u00a0", " ").strip()
    previous = None
    while previous != text:  # Wrappers can be nested or mixed: '"C:\a.mp4"'
        previous = text
        text = text.strip().strip(_WRAPPERS).strip()
    if text.lower().startswith("file:"):
        parsed = urlparse(text)
        text = unquote(parsed.path)
        if parsed.netloc and parsed.netloc.lower() != "localhost":
            text = f"//{parsed.netloc}{text}"  # file://server/share -> //server/share
        if windows and len(text) >= 3 and text[0] == "/" and text[2] == ":":
            text = text[1:]  # /C:/Videos -> C:/Videos
    if windows:
        for prefix, replacement in (("\\\\?\\UNC\\", "\\\\"), ("\\\\?\\", ""), ("//?/UNC/", "//"), ("//?/", "")):
            if text.startswith(prefix):
                text = replacement + text[len(prefix):]
                break
        text = "".join(ch for ch in text if ch not in _WINDOWS_INVALID)
    else:
        text = text.replace('"', "")
    return text.strip()


_PICKER_SCRIPT = r"""
import sys, tkinter
from tkinter import filedialog
kind, initial = sys.argv[1], sys.argv[2] or None
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)  # open in front of the browser
if kind == "folder":
    chosen = filedialog.askdirectory(parent=root, initialdir=initial, title="Chọn thư mục")
else:
    chosen = filedialog.askopenfilename(
        parent=root, initialdir=initial, title="Chọn video",
        filetypes=[("Video", "*.mp4 *.mkv *.avi *.mov *.ts *.webm"), ("Tất cả tệp", "*.*")])
root.destroy()
sys.stdout.buffer.write((chosen or "").encode("utf-8"))
"""


def pick_local_path(kind: str = "file", initial_dir: str | None = None, *, timeout: float = 600) -> str | None:
    """Open the operating system's file/folder dialog and return the chosen path.

    Only meaningful when the browser and this server run on the same computer:
    the dialog opens on the server's screen. Runs in a child process so Tk owns
    its own main thread instead of a Gradio worker thread.
    """
    if kind not in {"file", "folder"}:
        raise ValueError("kind phải là file hoặc folder")
    result = subprocess.run(
        [sys.executable, "-c", _PICKER_SCRIPT, kind, initial_dir or ""],
        capture_output=True, timeout=timeout, check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
        raise RuntimeError(f"Không mở được hộp thoại chọn tệp: {detail[-1] if detail else result.returncode}")
    chosen = result.stdout.decode("utf-8", errors="replace").strip()
    return os.path.normpath(chosen) if chosen else None
