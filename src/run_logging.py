"""Per-run file logging under repo logs/ (or ALCHEMIST_LOG_DIR)."""

from __future__ import annotations

import atexit
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_LOG_HANDLER: logging.FileHandler | None = None
_LOG_PATH: Path | None = None


def _truthy_env(name: str) -> bool:
    raw = os.environ.get(name, "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


class _UtcFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z"


def configure_run_logging(script_name: str) -> Path | None:
    """
    Attach a DEBUG FileHandler to the root logger. Idempotent per process.
    Set ALCHEMIST_NO_LOG_FILE=1 to disable. ALCHEMIST_LOG_DIR overrides output directory.
    """
    global _LOG_HANDLER, _LOG_PATH
    if _truthy_env("ALCHEMIST_NO_LOG_FILE"):
        return None
    if _LOG_HANDLER is not None:
        return _LOG_PATH

    raw_dir = os.environ.get("ALCHEMIST_LOG_DIR", "").strip()
    log_dir = Path(raw_dir).expanduser().resolve() if raw_dir else (REPO_ROOT / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = log_dir / f"{script_name}-{ts}-{os.getpid()}.log"

    fmt = _UtcFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)

    # Cap noisy libraries: compiler IR / failed optional FFmpeg shims are not app errors.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("torio").setLevel(logging.WARNING)

    _LOG_HANDLER = fh
    _LOG_PATH = path
    atexit.register(logging.shutdown)

    prev_hook = sys.excepthook

    def _log_excepthook(
        exc_type: type[BaseException], exc: BaseException, tb: object
    ) -> None:
        try:
            logging.getLogger("alchemist").critical(
                "Uncaught exception", exc_info=(exc_type, exc, tb)
            )
            if _LOG_HANDLER is not None:
                _LOG_HANDLER.flush()
        finally:
            prev_hook(exc_type, exc, tb)

    sys.excepthook = _log_excepthook
    return path


def log_process_banner(log: logging.Logger) -> None:
    log.debug("executable=%s", sys.executable)
    log.debug("python_version=%s", sys.version.replace("\n", " "))
    log.debug("platform=%s", platform.platform())
    log.debug("cwd=%s", Path.cwd())
    log.debug("argv=%s", sys.argv)


def log_alchemist_env(log: logging.Logger) -> None:
    """Log ALCHEMIST_* env and token presence (never values)."""
    lines: list[str] = []
    for k in sorted(os.environ):
        if not k.startswith("ALCHEMIST_"):
            continue
        v = os.environ[k]
        if len(v) > 1200:
            v = v[:1200] + "…(truncated)"
        lines.append(f"  {k}={v!r}")
    log.debug("ALCHEMIST_* environment (%d vars):\n%s", len(lines), "\n".join(lines) or "  (none)")
    hub_tok = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    log.debug("huggingface_token_in_env=%s", "yes" if hub_tok else "no")
    p = os.environ.get("PATH", "")
    log.debug("PATH_component_count=%d", len(p.split(os.pathsep)) if p else 0)


def flush_run_log() -> None:
    if _LOG_HANDLER is not None:
        _LOG_HANDLER.flush()
