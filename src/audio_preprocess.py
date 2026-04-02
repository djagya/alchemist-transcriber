"""Normalize arbitrary audio to 16 kHz mono PCM WAV via ffmpeg (session + enrollment clips)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PIPELINE_SAMPLE_RATE_HZ = 16000


def skip_ffmpeg_normalize() -> bool:
    return os.environ.get("ALCHEMIST_SKIP_FFMPEG_NORMALIZE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def normalize_to_pipeline_wav(
    src: Path,
    *,
    temp_prefix: str = "alchemist-audio-",
    quiet: bool = False,
) -> tuple[Path, Path | None]:
    """Transcode *src* to a temp 16 kHz mono s16le WAV when ffmpeg is available.

    Returns ``(path_to_use, temp_path_or_none)``. If *temp_path* is set, the caller
    must delete it when done. On skip/failure, returns ``(src, None)``.
    """
    if skip_ffmpeg_normalize():
        return src, None
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if not quiet:
            print(
                "warning: ffmpeg not on PATH; using original file. "
                "Install ffmpeg for reliable input handling (.qta, some m4a/mp4, etc.).",
                file=sys.stderr,
            )
        return src, None

    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix=temp_prefix)
    os.close(fd)
    tmp = Path(tmp_path)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ar",
        str(PIPELINE_SAMPLE_RATE_HZ),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError) as e:
        err = ""
        if isinstance(e, subprocess.CalledProcessError) and e.stderr:
            err = e.stderr.strip()[-1500:]
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        if not quiet:
            print(
                f"warning: ffmpeg normalize failed ({e}); using original file. {err}",
                file=sys.stderr,
            )
        return src, None
    return tmp, tmp
