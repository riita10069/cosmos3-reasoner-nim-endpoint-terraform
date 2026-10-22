from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_media_lock_root(
    media_lock: dict[str, Any], media_lock_path: Path
) -> Path:
    return Path(
        media_lock.get("media_root", media_lock_path.parent)
    ).expanduser().resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run(
    command: list[str],
    *,
    check: bool = True,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def command_version(command: list[str]) -> str | None:
    try:
        result = run(command, check=False)
    except FileNotFoundError:
        return None
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else None


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def environment_metadata(repo_root: Path) -> dict[str, Any]:
    git_commit = run(
        ["git", "rev-parse", "HEAD"], check=False, cwd=repo_root
    ).stdout.strip()
    git_dirty = bool(
        run(["git", "status", "--porcelain"], check=False, cwd=repo_root).stdout
    )
    return {
        "captured_at": utc_now(),
        "git_commit": git_commit or None,
        "git_dirty": git_dirty,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "ffmpeg": command_version(["ffmpeg", "-version"]),
        "ffprobe": command_version(["ffprobe", "-version"]),
        "opencv": _opencv_version(),
        "numpy": _numpy_version(),
        "aws_profile_set": bool(os.environ.get("AWS_PROFILE")),
    }


def _opencv_version() -> str | None:
    try:
        import cv2
    except ImportError:
        return None
    return cv2.__version__


def _numpy_version() -> str | None:
    try:
        import numpy
    except ImportError:
        return None
    return numpy.__version__
