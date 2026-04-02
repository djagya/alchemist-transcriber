"""Resolve enrollment directory and per-speaker reference clips (shared by CLI tools)."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ENROLLMENTS_DIRNAME = "enrollments"

# Prefer uncompressed / widely supported formats first.
ENROLLMENT_AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".wav",
    ".m4a",
    ".mp3",
    ".flac",
    ".ogg",
)


def _ext_rank(suffix: str) -> int:
    s = suffix.lower()
    for i, e in enumerate(ENROLLMENT_AUDIO_EXTENSIONS):
        if s == e.lower():
            return i
    return len(ENROLLMENT_AUDIO_EXTENSIONS)


def resolve_role_clip(enrollment_dir: Path, role_stem: str) -> Path | None:
    """Find danil.* / therapist.* (case-insensitive stem) with a supported extension."""
    role_l = role_stem.lower()
    for ext in ENROLLMENT_AUDIO_EXTENSIONS:
        for name in (f"{role_stem}{ext}", f"{role_l}{ext}"):
            p = enrollment_dir / name
            if p.is_file():
                return p
    allowed = {e.lower() for e in ENROLLMENT_AUDIO_EXTENSIONS}
    matches: list[Path] = []
    try:
        for p in enrollment_dir.iterdir():
            if not p.is_file():
                continue
            if p.stem.lower() != role_l:
                continue
            if p.suffix.lower() in allowed:
                matches.append(p)
    except OSError:
        return None
    if not matches:
        return None
    matches.sort(key=lambda x: (_ext_rank(x.suffix), x.name.lower()))
    return matches[0]


def resolve_enrollment_directory() -> tuple[Path | None, str]:
    """Pick enrollment directory and how it was chosen.

    Returns:
        (path, source) where source is:
        - ``env`` — ``ALCHEMIST_ENROLLMENT_DIR`` set to an existing directory
        - ``cwd_enrollments`` — ``./enrollments`` under the current working directory
        - ``disabled`` — env explicitly turns enrollment off
        - ``none`` — no directory (missing env path or no ``./enrollments``)
    """
    raw = os.environ.get("ALCHEMIST_ENROLLMENT_DIR")
    if raw is not None:
        s = raw.strip()
        if s.lower() in ("none", "off", "false", "0"):
            return None, "disabled"
        if s:
            p = Path(s).expanduser().resolve()
            return (p, "env") if p.is_dir() else (None, "none")
    auto = (Path.cwd() / DEFAULT_ENROLLMENTS_DIRNAME).resolve()
    if auto.is_dir():
        return auto, "cwd_enrollments"
    return None, "none"
