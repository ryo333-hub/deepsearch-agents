"""Windows-safe path validation for per-session host file access."""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath


PATH_ACCESS_DENIED_MESSAGE = "文件路径超出当前会话目录，不允许访问。"
INVALID_THREAD_ID_MESSAGE = "thread_id 参数无效。"
INVALID_UPLOAD_FILENAME_MESSAGE = "上传文件名无效。"

_THREAD_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,128}")
_WINDOWS_INVALID_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def validate_thread_id(thread_id: object) -> str:
    """Return a valid session identifier or reject it without normalization."""
    if not isinstance(thread_id, str) or not _THREAD_ID_PATTERN.fullmatch(thread_id):
        raise ValueError(INVALID_THREAD_ID_MESSAGE)
    return thread_id


def validate_upload_filename(filename: object) -> str:
    """Accept a plain Windows-safe filename without any directory component."""
    if not isinstance(filename, str) or not filename or len(filename) > 255:
        raise ValueError(INVALID_UPLOAD_FILENAME_MESSAGE)
    if filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError(INVALID_UPLOAD_FILENAME_MESSAGE)
    if not _is_windows_safe_component(filename):
        raise ValueError(INVALID_UPLOAD_FILENAME_MESSAGE)
    return filename


def resolve_session_directory(root: str | Path, thread_id: object) -> Path:
    """Build a validated ``session_<thread_id>`` directory under a trusted root."""
    safe_thread_id = validate_thread_id(thread_id)
    root_path = Path(root).resolve(strict=False)
    session_path = (root_path / f"session_{safe_thread_id}").resolve(strict=False)
    _require_containment(session_path, root_path)
    return session_path


def resolve_path(filename: object, session_dir: str | Path | None = None) -> str:
    """Resolve a user or Agent supplied relative path inside ``session_dir``.

    Absolute paths, drive-relative paths, UNC/device paths, Windows alternate
    data streams and invalid Windows components are rejected before any file
    read, write or directory creation occurs.
    """
    return str(_resolve_within_session(filename, session_dir))


def _resolve_within_session(
    filename: object,
    session_dir: str | Path | None,
) -> Path:
    if not isinstance(filename, str) or not filename or "\x00" in filename:
        raise ValueError(PATH_ACCESS_DENIED_MESSAGE)
    if session_dir is None or not str(session_dir):
        raise ValueError(PATH_ACCESS_DENIED_MESSAGE)

    normalized = filename.replace("\\", "/")
    windows_path = PureWindowsPath(filename)
    if (
        normalized.startswith("/")
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
    ):
        raise ValueError(PATH_ACCESS_DENIED_MESSAGE)

    components = normalized.split("/")
    if any(not component or not _is_windows_safe_component(component) for component in components):
        raise ValueError(PATH_ACCESS_DENIED_MESSAGE)

    base = Path(session_dir).resolve(strict=False)
    target = (base / Path(*components)).resolve(strict=False)
    _require_containment(target, base)
    return target


def _is_windows_safe_component(component: str) -> bool:
    """Reject Windows path metacharacters, ADS syntax and device names."""
    if component == ".":
        return True
    if component == "..":
        return False
    if component.endswith((" ", ".")):
        return False
    if any(ord(char) < 32 or char in _WINDOWS_INVALID_CHARS for char in component):
        return False
    reserved_candidate = component.split(".", 1)[0].upper()
    return reserved_candidate not in _WINDOWS_RESERVED_NAMES


def _require_containment(target: Path, base: Path) -> None:
    """Reject targets outside ``base`` after symlink/junction resolution."""
    if target != base and not target.is_relative_to(base):
        raise ValueError(PATH_ACCESS_DENIED_MESSAGE)
