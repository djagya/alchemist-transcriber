"""
ASR + diarization + speaker labels → markdown.
CLI: --input path --output path.md (see README for enrollment).

Env:
  ALCHEMIST_PIPELINE — mlx (default) | torch — mlx: mlx-whisper + diarize package; torch: openai-whisper + pyannote (needs uv sync --extra torch-pipeline, HF_TOKEN).
  ALCHEMIST_INITIAL_PROMPT — optional ASR context (names, domain terms); overrides built-in default.
  ALCHEMIST_LANGUAGE — Whisper language code (e.g. ru, en). If env var is unset → default ru.
    Set to auto (or empty value in .env) to use Whisper auto-detect instead.
  ALCHEMIST_NUM_SPEAKERS — default 2; set to "auto" for automatic speaker count (mlx diarize; passed to pyannote when supported).
  ALCHEMIST_ENROLLMENT_DIR — optional override path; if unset, uses ./enrollments when that folder exists.
    Set to none/off/false/0 to skip enrollment even if ./enrollments exists.
    Clips: danil.* and therapist.* with extensions .wav, .m4a, .mp3, .flac, .ogg.
  ALCHEMIST_SKIP_FFMPEG_NORMALIZE — if 1/true, do not run ffmpeg; use the original file as-is.
  ALCHEMIST_WHISPER_REPO — MLX Hub repo (mlx pipeline only; default mlx-community/whisper-large-v3-mlx).
    Smaller repos (e.g. whisper-medium-mlx, whisper-small-mlx) use less RAM at some accuracy cost.
  ALCHEMIST_WORD_TIMESTAMPS — mlx pipeline: 1 (default) | 0 — set 0 to disable per-word timestamps
    (lower MLX memory; transcript still uses segment-level timing from Whisper).
  ALCHEMIST_MLX_CONDITION_ON_PREVIOUS_TEXT — mlx: 1 | 0 (default) — if 1, Whisper passes prior decoded
    text into the next window (may help names/context; can revive repetition hallucinations and slow decode).
  ALCHEMIST_MLX_NO_SPEECH_THRESHOLD — optional float; mlx only; mlx_whisper default 0.6.
  ALCHEMIST_MLX_LOGPROB_THRESHOLD — optional float; mlx only; default -1.0.
  ALCHEMIST_MLX_COMPRESSION_RATIO_THRESHOLD — optional float; mlx only; default 2.4.
  ALCHEMIST_MLX_HALLUCINATION_SILENCE_THRESHOLD — optional float; mlx only (mlx_whisper default unset).
  ALCHEMIST_VAD_SPEECH_PAD_MS — Silero get_speech_timestamps speech_pad_ms (mlx VAD prefilter; default 200).
  ALCHEMIST_VAD_TAIL_PRESERVE_SEC — last N seconds of audio passed to Whisper unchanged after VAD mask
    (captures quiet afterthoughts Silero misses; default 25; set 0 to disable).
  ALCHEMIST_MLX_TAIL_RETRANSCRIBE — mlx only: 1 (default) | 0 — second Whisper pass on the last
    ALCHEMIST_VAD_TAIL_PRESERVE_SEC of normalized audio to recover lines dropped on the full pass.
  ALCHEMIST_SCRUB_REPEAT_TOKENS — 1 (default) | 0 — remove trailing/internal runs of repeated 1-letter
    ASR junk (e.g. «в в в» on silence) before writing markdown.
  ALCHEMIST_ASR_TEXT_FIXES — 1 (default) | 0 — hyphen spacing (что-то, из-за), name/typo replacements (see transcript_fixes).
  ALCHEMIST_OPENAI_WHISPER_MODEL — openai-whisper model name (torch pipeline only; default large-v3).
  ALCHEMIST_WHISPER_DEVICE — torch ASR device: auto | cpu | mps | cuda (torch pipeline; auto uses CPU on macOS because openai-whisper is not MPS-safe).
  ALCHEMIST_DIARIZATION_DEVICE — pyannote device: cpu | mps | cuda (torch pipeline; default cpu).
  ALCHEMIST_QUIET — 1 | 0 (default) — same as CLI --quiet: suppress ffmpeg warnings on stderr only.
  ALCHEMIST_VERBOSE_META — 1 | 0 (default) — same as CLI --verbose: full YAML frontmatter (debug/diagnostic
    fields). Default output uses a short frontmatter only.
  ALCHEMIST_NO_LOG_FILE — 1 | 0 (default) — disable per-run debug log file under logs/ (or ALCHEMIST_LOG_DIR).
  ALCHEMIST_LOG_DIR — optional absolute directory for run log files (default: <repo>/logs).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import importlib.metadata
import os
import re
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from dotenv import load_dotenv

from audio_preprocess import PIPELINE_SAMPLE_RATE_HZ, normalize_to_pipeline_wav
from diar_types import DiarSeg
from enrollment_util import resolve_enrollment_directory, resolve_role_clip
from transcript_fixes import apply_transcript_text_fixes, asr_text_fixes_enabled
from run_logging import (
    configure_run_logging,
    flush_run_log,
    log_alchemist_env,
    log_process_banner,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOG = logging.getLogger("alchemist.transcribe")

DEFAULT_INITIAL_PROMPT = (
    "Психотерапевтическая беседа на русском, возможны вкрапления английского. "
    "Участники: клиент и терапевт. Имена собственные: Тея (склонения: Теи, Тее, Теей, с Теей), Лара. "
    "Темы: отношения, любовь, ревность, близость, сексуальность, семья, родители, детство, "
    "работа, деньги, смысл, смерть, одиночество, стыд, вина, злость, тревога, депрессия, "
    "границы, контроль, манипуляция, насилие, агрессия, самоуважение. "
    "Про жизнь и опыт: наркотики, алкоголь, употребление, зависимость, трезвость, срыв, "
    "отход, ломка, триггер, токсикология, психоз, ПАВ. "
    "Термины: терапевт, клиент, терапия, трансфер, контрперенос. "
    "Therapy session; Danil and Therapist; drugs, addiction, life topics, psychology."
)
DEFAULT_WHISPER_LANGUAGE = "ru"
DEFAULT_WHISPER_REPO = "mlx-community/whisper-large-v3-mlx"
DEFAULT_NAMES = ("Danil", "Therapist")
INTERRUPTION_GAP_SEC = 0.4


def _whisper_repo() -> str:
    raw = os.environ.get("ALCHEMIST_WHISPER_REPO", DEFAULT_WHISPER_REPO).strip()
    return raw or DEFAULT_WHISPER_REPO


def _mlx_word_timestamps() -> bool:
    raw = os.environ.get("ALCHEMIST_WORD_TIMESTAMPS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _mlx_condition_on_previous_text() -> bool:
    raw = os.environ.get("ALCHEMIST_MLX_CONDITION_ON_PREVIOUS_TEXT", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _optional_env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _mlx_decode_overrides_from_env() -> tuple[dict[str, Any], dict[str, float]]:
    """Optional mlx_whisper.transcribe kwargs from env; second dict is for transcript metadata."""
    pairs = (
        ("ALCHEMIST_MLX_NO_SPEECH_THRESHOLD", "no_speech_threshold"),
        ("ALCHEMIST_MLX_LOGPROB_THRESHOLD", "logprob_threshold"),
        ("ALCHEMIST_MLX_COMPRESSION_RATIO_THRESHOLD", "compression_ratio_threshold"),
        (
            "ALCHEMIST_MLX_HALLUCINATION_SILENCE_THRESHOLD",
            "hallucination_silence_threshold",
        ),
    )
    kw: dict[str, Any] = {}
    meta: dict[str, float] = {}
    for env_key, arg in pairs:
        v = _optional_env_float(env_key)
        if v is not None:
            kw[arg] = v
            meta[arg] = v
    return kw, meta


def _pipeline_mode() -> str:
    m = os.environ.get("ALCHEMIST_PIPELINE", "mlx").strip().lower()
    if m in ("torch", "pytorch", "openai"):
        return "torch"
    return "mlx"


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _openai_whisper_model() -> str:
    raw = os.environ.get("ALCHEMIST_OPENAI_WHISPER_MODEL", "large-v3").strip()
    return raw or "large-v3"


def _audio_duration_sf(path: Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return 0.0


_ws_model: Any = None


def _wespeaker():
    global _ws_model
    if _ws_model is None:
        import wespeakerruntime as wespeaker_rt

        _ws_model = wespeaker_rt.Speaker(lang="en")
    return _ws_model


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_frontmatter_sha256(md_path: Path) -> str | None:
    if not md_path.is_file():
        return None
    text = md_path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end == -1:
        return None
    block = text[4:end]
    try:
        meta = yaml.safe_load(block) or {}
    except yaml.YAMLError:
        return None
    h = meta.get("source_sha256")
    return h if isinstance(h, str) and len(h) == 64 else None


def _alchemist_language() -> str | None:
    """Whisper `language` decode option. Missing env → Russian; `auto` or empty → detect."""
    if "ALCHEMIST_LANGUAGE" not in os.environ:
        return DEFAULT_WHISPER_LANGUAGE
    raw = os.environ.get("ALCHEMIST_LANGUAGE", "").strip().lower()
    if raw in ("", "auto"):
        return None
    return raw


def _num_speakers_arg() -> int | None:
    raw = os.environ.get("ALCHEMIST_NUM_SPEAKERS", "2").strip().lower()
    if raw in ("auto", ""):
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n >= 1 else None


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _speaker_for_interval(
    start: float, end: float, diar: list[DiarSeg], mid_fallback: float | None = None
) -> str:
    if not diar:
        return "SPEAKER_00"
    best_label: str | None = None
    best_ov = 0.0
    for d in diar:
        ov = _overlap(start, end, d.start, d.end)
        if ov > best_ov:
            best_ov, best_label = ov, d.label
    if best_label is not None and best_ov > 0:
        return best_label
    mid = mid_fallback if mid_fallback is not None else 0.5 * (start + end)
    for d in diar:
        if d.start <= mid <= d.end:
            return d.label
    nearest = min(diar, key=lambda d: min(abs(mid - d.start), abs(mid - d.end)))
    return nearest.label


def _chronological_label_map(diar: list[DiarSeg]) -> dict[str, str]:
    seen: list[str] = []
    for d in sorted(diar, key=lambda x: (x.start, x.end)):
        if d.label not in seen:
            seen.append(d.label)
    mapping: dict[str, str] = {}
    for i, lab in enumerate(seen):
        if i < len(DEFAULT_NAMES):
            mapping[lab] = DEFAULT_NAMES[i]
        else:
            mapping[lab] = f"Speaker {i + 1}"
    return mapping


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _embedding_from_path(path: Path) -> np.ndarray | None:
    try:
        emb = _wespeaker().extract_embedding(str(path))
        return np.asarray(emb, dtype=np.float64).reshape(-1)
    except Exception:
        return None


def _embedding_from_time_range(path: Path, start: float, end: float) -> np.ndarray | None:
    start = max(0.0, start)
    end = max(start + 0.05, end)
    tmp_path: str | None = None
    try:
        data, sr = sf.read(str(path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        s0 = int(start * sr)
        s1 = min(len(data), int(end * sr))
        chunk = data[s0:s1]
        min_samples = max(3200, int(0.15 * sr))
        if len(chunk) < min_samples:
            return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, chunk, sr)
        return _embedding_from_path(Path(tmp_path))
    except Exception:
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _first_start_for_label(diar: list[DiarSeg], label: str) -> float:
    return min(d.start for d in diar if d.label == label)


def _wespeaker_embedding_label_map(
    diar: list[DiarSeg], enrollment_dir: Path, main_audio: Path
) -> dict[str, str] | None:
    danil_src = resolve_role_clip(enrollment_dir, "danil")
    therapist_src = resolve_role_clip(enrollment_dir, "therapist")
    if danil_src is None or therapist_src is None:
        return None

    danil_w, danil_tmp = normalize_to_pipeline_wav(
        danil_src, temp_prefix="alchemist-enroll-d-", quiet=True
    )
    therapist_w, therapist_tmp = normalize_to_pipeline_wav(
        therapist_src, temp_prefix="alchemist-enroll-t-", quiet=True
    )
    try:
        ref_d = _embedding_from_path(danil_w)
        ref_t = _embedding_from_path(therapist_w)
        if ref_d is None or ref_t is None:
            return None

        by_label: dict[str, list[DiarSeg]] = {}
        for d in diar:
            by_label.setdefault(d.label, []).append(d)

        centroids: dict[str, np.ndarray] = {}
        for lab, segs in by_label.items():
            segs_sorted = sorted(segs, key=lambda s: s.end - s.start, reverse=True)
            embs: list[np.ndarray] = []
            for s in segs_sorted[:6]:
                if s.end - s.start < 0.2:
                    continue
                crop_end = min(s.end, s.start + 4.0)
                e = _embedding_from_time_range(main_audio, s.start, crop_end)
                if e is not None:
                    embs.append(e)
            if embs:
                centroids[lab] = np.mean(np.stack(embs, axis=0), axis=0)

        if not centroids:
            return None

        cluster_labels = sorted(
            centroids.keys(), key=lambda lab: _first_start_for_label(diar, lab)
        )
        assign_lab_to_role: dict[str, str] = {}

        if len(cluster_labels) >= 2:
            a, b = cluster_labels[0], cluster_labels[1]
            ca, cb = centroids[a], centroids[b]
            score_a_d_t = _cosine(ca, ref_d) + _cosine(cb, ref_t)
            score_a_t_d = _cosine(ca, ref_t) + _cosine(cb, ref_d)
            if score_a_d_t >= score_a_t_d:
                assign_lab_to_role[a], assign_lab_to_role[b] = (
                    DEFAULT_NAMES[0],
                    DEFAULT_NAMES[1],
                )
            else:
                assign_lab_to_role[a], assign_lab_to_role[b] = (
                    DEFAULT_NAMES[1],
                    DEFAULT_NAMES[0],
                )
            for i, lab in enumerate(cluster_labels[2:], start=3):
                assign_lab_to_role[lab] = f"Speaker {i}"
        else:
            lab0 = cluster_labels[0]
            if _cosine(centroids[lab0], ref_d) >= _cosine(centroids[lab0], ref_t):
                assign_lab_to_role[lab0] = DEFAULT_NAMES[0]
            else:
                assign_lab_to_role[lab0] = DEFAULT_NAMES[1]

        mapping: dict[str, str] = {}
        extra_idx = 0
        for lab in sorted(by_label.keys(), key=lambda x: _first_start_for_label(diar, x)):
            if lab in assign_lab_to_role:
                mapping[lab] = assign_lab_to_role[lab]
            else:
                extra_idx += 1
                mapping[lab] = f"Speaker {len(DEFAULT_NAMES) + extra_idx}"
        return mapping
    finally:
        for p in (danil_tmp, therapist_tmp):
            if p is not None:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass


def _vad_speech_pad_ms() -> int:
    raw = os.environ.get("ALCHEMIST_VAD_SPEECH_PAD_MS", "200").strip()
    try:
        n = int(raw)
    except ValueError:
        return 200
    return max(0, min(n, 2000))


def _vad_tail_preserve_seconds() -> float:
    """Trailing seconds of original audio to keep for Whisper after VAD (quiet outros)."""
    raw = os.environ.get("ALCHEMIST_VAD_TAIL_PRESERVE_SEC")
    if raw is None:
        return 25.0
    s = raw.strip().lower()
    if s in ("0", "false", "no", "off", ""):
        return 0.0
    try:
        return max(0.0, float(s))
    except ValueError:
        return 25.0


def _vad_filter_audio(
    audio_path: Path,
) -> tuple[Path | None, int, int, float]:
    """Run Silero VAD and zero out non-speech segments.

    Returns (path_to_filtered_temp_wav, speech_segment_count, speech_pad_ms_used,
    tail_preserve_sec_used). If VAD finds nothing or fails, returns (None, 0, pad, tail).
    Caller must delete the temp file when done.
    """
    pad_ms = _vad_speech_pad_ms()
    tail_sec = _vad_tail_preserve_seconds()
    _LOG.debug(
        "VAD prefilter start path=%s speech_pad_ms=%s tail_preserve_sec=%s",
        audio_path,
        pad_ms,
        tail_sec,
    )
    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as e:
        _LOG.warning(
            "VAD prefilter skipped (silero/torch unavailable): %s — Whisper uses full normalized wav",
            e,
        )
        return None, 0, pad_ms, tail_sec

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    wav_t: Any = None
    model: Any = None
    try:
        wav_t = torch.from_numpy(audio).float()
        model = load_silero_vad()
        speech_ts = get_speech_timestamps(
            wav_t, model, sampling_rate=sr, speech_pad_ms=pad_ms
        )
        if not speech_ts:
            _LOG.info("VAD: no speech regions; Whisper input is full normalized audio")
            return None, 0, pad_ms, tail_sec

        filtered = np.zeros_like(audio)
        for ts in speech_ts:
            s, e = ts["start"], min(ts["end"], len(audio))
            filtered[s:e] = audio[s:e]

        if tail_sec > 0.0:
            n_tail = min(len(audio), int(round(tail_sec * float(sr))))
            if n_tail > 0:
                filtered[-n_tail:] = audio[-n_tail:]

        fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="alchemist-vad-")
        os.close(fd)
        sf.write(tmp_name, filtered, sr, subtype="PCM_16")
        p = Path(tmp_name)
        _LOG.info(
            "VAD: masked wav written path=%s speech_segments=%d audio_samples=%d sr=%d",
            p,
            len(speech_ts),
            len(audio),
            int(sr),
        )
        return p, len(speech_ts), pad_ms, tail_sec
    finally:
        if wav_t is not None:
            del wav_t
        if model is not None:
            del model
        del audio
        gc.collect()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _is_hallucinated_segment(text: str) -> bool:
    """Detect repetitive hallucination patterns in a transcript segment."""
    words = text.split()
    if len(words) < 4:
        return False
    lower = [w.lower().strip(".,!?;:") for w in words]
    unique = set(lower)
    if len(unique) <= 2 and len(words) >= 6:
        return True
    if len(unique) / len(words) < 0.15 and len(words) > 8:
        return True
    for ngram_len in (1, 2, 3):
        from collections import Counter

        ngrams = [
            " ".join(lower[i : i + ngram_len])
            for i in range(len(lower) - ngram_len + 1)
        ]
        if not ngrams:
            continue
        most_common_count = Counter(ngrams).most_common(1)[0][1]
        if most_common_count / len(ngrams) > 0.6 and len(words) > 6:
            return True
    return False


def _mlx_tail_retranscribe_enabled() -> bool:
    raw = os.environ.get("ALCHEMIST_MLX_TAIL_RETRANSCRIBE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _is_subtitle_tail_hallucination(text: str) -> bool:
    t = text.lower().strip()
    needles = (
        "продолжение следует",
        "спасибо за субтитры",
        "субтитры",
        "thanks for watching",
        "amara.org",
    )
    return any(n in t for n in needles)


def _segment_time_overlap_ratio(
    st: float, en: float, main_segs: list[dict]
) -> float:
    """Max overlap of [st,en] with any main segment / length of [st,en]."""
    span = max(1e-6, en - st)
    best = 0.0
    for m in main_segs:
        m0, m1 = float(m.get("start", 0)), float(m.get("end", 0))
        ov = _overlap(st, en, m0, m1)
        best = max(best, ov / span)
    return best


def _mlx_sanitize_segments_before_tail_merge(
    asr_result: dict, *, duration: float
) -> dict:
    """Drop subtitle-style hallucinations and long bogus segments at file end.

    Full-file Whisper can invent a long closing segment that overlaps the real
    quiet outro; a tail-only pass then refuses to merge. Removing those blobs
    lets short genuine tail segments attach.
    """
    kept: list[dict] = []
    for s in asr_result.get("segments") or []:
        text = (s.get("text") or "").strip()
        if _is_subtitle_tail_hallucination(text):
            continue
        st, en = float(s.get("start", 0.0)), float(s.get("end", 0.0))
        if duration > 0 and en >= duration - 3.0 and (en - st) >= 9.0:
            continue
        kept.append(s)
    out = dict(asr_result)
    out["segments"] = kept
    return out


def _mlx_merge_tail_retranscribe(
    work_inp: Path,
    asr_result: dict,
    *,
    tail_sec: float,
    whisper_kw: dict[str, Any],
) -> tuple[dict, int]:
    """Re-transcribe the last *tail_sec* of normalized audio and merge missing segments.

    Full-file Whisper sometimes drops very quiet closing lines; an isolated tail pass
    often recovers them. Segments that mostly overlap the main pass are skipped.

    Returns (possibly updated asr_result, number of segments appended).
    """
    if tail_sec <= 0.0 or not _mlx_tail_retranscribe_enabled():
        _LOG.debug(
            "MLX tail retranscribe skipped (tail_sec=%s enabled=%s)",
            tail_sec,
            _mlx_tail_retranscribe_enabled(),
        )
        return asr_result, 0
    try:
        import mlx_whisper
    except ImportError as e:
        _LOG.warning("MLX tail retranscribe skipped (mlx_whisper missing): %s", e)
        return asr_result, 0

    _LOG.debug("MLX tail retranscribe: reading work_inp=%s tail_sec=%s", work_inp, tail_sec)
    audio, sr = sf.read(str(work_inp), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = float(len(audio)) / float(sr)
    if duration <= tail_sec + 1.0:
        _LOG.debug(
            "MLX tail retranscribe skipped: duration=%.3fs too short for tail_sec=%s",
            duration,
            tail_sec,
        )
        return asr_result, 0

    offset = duration - tail_sec
    i0 = int(round(offset * float(sr)))
    chunk = audio[i0:]
    if len(chunk) < int(0.5 * sr):
        return asr_result, 0

    fd, tmp_name = tempfile.mkstemp(suffix=".wav", prefix="alchemist-tail-asr-")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        sf.write(str(tmp_path), chunk, sr, subtype="PCM_16")
        tw: dict[str, Any] = dict(whisper_kw)
        tw["word_timestamps"] = False
        tail_result = mlx_whisper.transcribe(str(tmp_path), **tw)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    main_segs = list(asr_result.get("segments") or [])
    main_end = max((float(s.get("end", 0.0)) for s in main_segs), default=0.0)

    added = 0
    for s in tail_result.get("segments") or []:
        text = (s.get("text") or "").strip()
        if not text or _is_subtitle_tail_hallucination(text):
            continue
        st = float(s["start"]) + offset
        en = float(s["end"]) + offset
        if en <= main_end - 0.15 and _segment_time_overlap_ratio(st, en, main_segs) >= 0.35:
            continue
        if _segment_time_overlap_ratio(st, en, main_segs) >= 0.65:
            continue
        ns = dict(s)
        ns["start"] = st
        ns["end"] = en
        if "words" in ns:
            del ns["words"]
        main_segs.append(ns)
        added += 1

    if added == 0:
        _LOG.debug("MLX tail retranscribe: no new segments merged from tail pass")
        return asr_result, 0
    main_segs.sort(key=lambda x: float(x["start"]))
    out = dict(asr_result)
    out["segments"] = main_segs
    _LOG.info("MLX tail retranscribe: merged %d segment(s) from tail-only pass", added)
    return out, added


def _filter_hallucinated_segments(asr_result: dict) -> tuple[dict, int]:
    """Remove segments with hallucinated repetitive text.

    Returns (cleaned_result, number_of_removed_segments).
    """
    segments = asr_result.get("segments") or []
    kept = []
    removed = 0
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if _is_hallucinated_segment(text):
            removed += 1
        else:
            kept.append(seg)
    cleaned = dict(asr_result)
    cleaned["segments"] = kept
    return cleaned, removed


def _count_interruptions(diar: list[DiarSeg]) -> int:
    if len(diar) < 2:
        return 0
    srt = sorted(diar, key=lambda x: (x.start, x.end))
    n = 0
    for i in range(1, len(srt)):
        prev, cur = srt[i - 1], srt[i]
        if cur.label == prev.label:
            continue
        gap = cur.start - prev.end
        if gap < 0 or (0 <= gap < INTERRUPTION_GAP_SEC):
            n += 1
    return n


def _scrub_repeat_tokens_enabled() -> bool:
    raw = os.environ.get("ALCHEMIST_SCRUB_REPEAT_TOKENS", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _word_alphanumeric_core(word: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", word, flags=re.UNICODE)


def _scrub_repeated_short_token_artifacts(text: str) -> str:
    """Drop ASR filler: 3+ repeats of the same 1-letter Russian token (в, и, а, …).

    Common when the model decodes noise/silence. Industry practice: regex/heuristic
    cleanup or LM rescoring; we use a conservative token run collapse.
    """
    if not text or not _scrub_repeat_tokens_enabled():
        return text

    parts = text.split()
    if not parts:
        return text

    def is_junk_letter(core: str) -> bool:
        if len(core) != 1:
            return False
        if not core.isalpha():
            return False
        o = ord(core.lower())
        # Latin or Cyrillic single letters only (avoid digits/punct)
        if "a" <= core.lower() <= "z":
            return True
        if 0x0430 <= o <= 0x044F or core.lower() == "ё":
            return True
        return False

    def strip_run_from_end(tok: list[str]) -> list[str]:
        while len(tok) >= 3:
            core = _word_alphanumeric_core(tok[-1]).lower()
            if not is_junk_letter(core):
                break
            run = 1
            i = len(tok) - 2
            while i >= 0 and _word_alphanumeric_core(tok[i]).lower() == core:
                run += 1
                i -= 1
            if run >= 3:
                tok = tok[: i + 1]
            else:
                break
        return tok

    def strip_run_from_start(tok: list[str]) -> list[str]:
        while len(tok) >= 3:
            core = _word_alphanumeric_core(tok[0]).lower()
            if not is_junk_letter(core):
                break
            run = 1
            i = 1
            while i < len(tok) and _word_alphanumeric_core(tok[i]).lower() == core:
                run += 1
                i += 1
            if run >= 3:
                tok = tok[i:]
            else:
                break
        return tok

    parts = strip_run_from_end(parts)
    parts = strip_run_from_start(parts)

    # Internal: collapse 5+ consecutive identical 1-letter junk to nothing
    out: list[str] = []
    i = 0
    while i < len(parts):
        core = _word_alphanumeric_core(parts[i]).lower()
        if is_junk_letter(core):
            j = i + 1
            while j < len(parts) and _word_alphanumeric_core(parts[j]).lower() == core:
                j += 1
            run = j - i
            if run >= 5:
                i = j
                continue
        out.append(parts[i])
        i += 1

    return " ".join(out).strip()


def _format_timestamp(seconds: float) -> str:
    assert seconds >= 0
    ms_total = int(round(seconds * 1000.0))
    h = ms_total // 3_600_000
    ms_total -= h * 3_600_000
    m = ms_total // 60_000
    ms_total -= m * 60_000
    s = ms_total // 1000
    ms = ms_total - s * 1000
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        suffix=".md.tmp", dir=str(path.parent), text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.write(content)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _transcribe_quiet() -> bool:
    raw = os.environ.get("ALCHEMIST_QUIET", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _transcribe_verbose_meta() -> bool:
    raw = os.environ.get("ALCHEMIST_VERBOSE_META", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _minimal_transcript_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Strip debug/diagnostic frontmatter keys for downstream tools (e.g. OpenClaw)."""
    keys = (
        "source_path",
        "source_sha256",
        "audio_duration_seconds",
        "word_count",
        "interruption_count",
        "language",
        "pipeline",
        "asr_model",
        "diarization_backend",
        "speaker_assignment_strategy",
        "speaker_label_map",
        "generated_at_utc",
    )
    return {k: meta[k] for k in keys if k in meta}


def _build_markdown(
    meta: dict[str, Any],
    blocks: list[tuple[str, float, float, str]],
) -> str:
    fm = yaml.safe_dump(
        meta,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
        width=4096,
    ).rstrip()
    lines = [f"---\n{fm}\n---\n", "\n## Transcript\n"]
    for name, t0, t1, text in blocks:
        ts0, ts1 = _format_timestamp(t0), _format_timestamp(t1)
        lines.append(f"\n**{name}** ({ts0}–{ts1})\n\n{text.strip()}\n")
    return "".join(lines).rstrip() + "\n"


def main() -> int:
    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv()

    log_path = configure_run_logging("transcribe")
    if log_path is not None:
        print(f"alchemist: full debug log: {log_path}", file=sys.stderr)
    log_process_banner(_LOG)
    log_alchemist_env(_LOG)

    parser = argparse.ArgumentParser(
        description="Whisper ASR + diarization → markdown transcript (MLX or PyTorch pipeline)."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input audio file")
    parser.add_argument(
        "--output", required=True, type=Path, help="Output .md path (any directory; created if needed)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress ffmpeg warnings on stderr (does not change YAML; default frontmatter is already short).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Write full YAML frontmatter (VAD stats, tail merge, package versions, enrollment paths, etc.).",
    )
    args = parser.parse_args()
    inp: Path = args.input.expanduser().resolve()
    out: Path = args.output.expanduser().resolve()
    quiet = bool(args.quiet) or _transcribe_quiet()
    verbose_meta = bool(args.verbose) or _transcribe_verbose_meta()
    _LOG.info(
        "cli: input=%s output=%s quiet=%s verbose_meta=%s",
        inp,
        out,
        quiet,
        verbose_meta,
    )

    if not inp.is_file():
        print(f"error: input not found: {inp}", file=sys.stderr)
        _LOG.error("input not found: %s", inp)
        flush_run_log()
        return 1

    digest = _sha256_file(inp)
    existing = _read_frontmatter_sha256(out)
    _LOG.debug(
        "idempotency check: existing_frontmatter_sha256=%s computed_input_sha256=%s",
        existing,
        digest,
    )
    if existing == digest:
        _LOG.info(
            "exit 0 (idempotent): output already matches input sha256=%s… out=%s",
            digest[:20],
            out,
        )
        flush_run_log()
        return 0

    _ip_env = os.environ.get("ALCHEMIST_INITIAL_PROMPT")
    if _ip_env is None:
        initial_prompt = DEFAULT_INITIAL_PROMPT
    else:
        initial_prompt = _ip_env.strip() or DEFAULT_INITIAL_PROMPT
    lang = _alchemist_language()
    num_speakers = _num_speakers_arg()
    _LOG.info(
        "decode options: language=%r num_speakers=%s initial_prompt_len=%d chars (env override=%s)",
        lang,
        num_speakers,
        len(initial_prompt),
        _ip_env is not None,
    )

    t_wall0 = time.perf_counter()

    _LOG.info("normalizing audio via ffmpeg (if needed) input=%s quiet=%s", inp, quiet)
    work_inp, tmp_normalized = normalize_to_pipeline_wav(inp, quiet=quiet)
    preprocess = (
        "none"
        if tmp_normalized is None
        else f"ffmpeg_pcm_s16le_{PIPELINE_SAMPLE_RATE_HZ}hz_mono_wav"
    )
    _LOG.info(
        "normalize result: work_inp=%s temp_normalized=%s preprocess=%s",
        work_inp,
        tmp_normalized,
        preprocess,
    )

    try:
        duration_audio = _audio_duration_sf(work_inp)
        mode = _pipeline_mode()
        _LOG.info("pipeline_mode=%s audio_duration_seconds=%.3f", mode, duration_audio)
        mlx_decode_meta: dict[str, float] = {}
        mlx_tail_segments_merged = 0
        mlx_condition_on_previous_text = False

        if mode == "torch":
            tok = _hf_token()
            if not tok:
                print(
                    "error: ALCHEMIST_PIPELINE=torch requires HF_TOKEN (or "
                    "HUGGING_FACE_HUB_TOKEN) for pyannote models.",
                    file=sys.stderr,
                )
                _LOG.error("torch pipeline: missing HF_TOKEN / HUGGING_FACE_HUB_TOKEN")
                flush_run_log()
                return 1
            try:
                from torch_pipeline import PYANNOTE_MODEL, run_openai_whisper_pyannote
            except ImportError as e:
                print(
                    "error: torch pipeline dependencies missing. Install with: "
                    "uv sync --extra torch-pipeline",
                    file=sys.stderr,
                )
                print(e, file=sys.stderr)
                _LOG.exception("torch pipeline: import error: %s", e)
                flush_run_log()
                return 1

            _LOG.info(
                "torch: starting ASR+diarization model=%s whisper_dev will be reported",
                _openai_whisper_model(),
            )
            asr_result, diar, dur2, whisper_dev, dia_dev = run_openai_whisper_pyannote(
                work_inp,
                hf_token=tok,
                initial_prompt=initial_prompt,
                language=lang,
                whisper_model=_openai_whisper_model(),
                num_speakers=num_speakers,
            )
            n_asr_seg = len(asr_result.get("segments") or [])
            _LOG.info(
                "torch: ASR+diarization done segments=%d diar_labels=%d whisper_dev=%s diar_dev=%s",
                n_asr_seg,
                len(diar),
                whisper_dev,
                dia_dev,
            )
            if duration_audio <= 0.0:
                duration_audio = dur2
            if duration_audio <= 0.0:
                ends = [
                    float(s.get("end", 0.0)) for s in (asr_result.get("segments") or [])
                ]
                duration_audio = max(ends) if ends else 0.0
            if not diar:
                _LOG.warning("torch: empty diarization; using single SPEAKER_00 span")
                diar = [DiarSeg(0.0, max(duration_audio, 1.0), "SPEAKER_00")]

            try:
                pyannote_ver = importlib.metadata.version("pyannote-audio")
            except importlib.metadata.PackageNotFoundError:
                pyannote_ver = "unknown"

            diarize_ver = pyannote_ver
            asr_model_meta = f"openai-whisper {_openai_whisper_model()}"
            diar_backend = "pyannote"
            diar_model_meta = PYANNOTE_MODEL
            asr_runtime_meta = f"pytorch ({whisper_dev})"
            diar_device_meta = dia_dev
            pipeline_meta = "torch"
        else:
            import mlx_whisper
            from diarize import diarize as run_diarize

            _LOG.info("mlx: loading mlx_whisper and diarize backend")
            vad_path, vad_seg_count, vad_pad_ms, vad_tail_sec = _vad_filter_audio(
                work_inp
            )
            whisper_input = str(vad_path) if vad_path else str(work_inp)
            _LOG.info(
                "mlx: whisper_input=%s (vad_temp=%s) vad_speech_segments=%s pad_ms=%s tail_preserve_sec=%s",
                whisper_input,
                vad_path is not None,
                vad_seg_count,
                vad_pad_ms,
                vad_tail_sec,
            )

            mlx_condition_on_previous_text = _mlx_condition_on_previous_text()
            whisper_kw: dict[str, Any] = {
                "path_or_hf_repo": _whisper_repo(),
                "initial_prompt": initial_prompt,
                "word_timestamps": _mlx_word_timestamps(),
                "verbose": False,
                "condition_on_previous_text": mlx_condition_on_previous_text,
            }
            mlx_decode_kw, mlx_decode_meta = _mlx_decode_overrides_from_env()
            whisper_kw.update(mlx_decode_kw)
            if lang:
                whisper_kw["language"] = lang

            safe_kw = {
                k: (f"<{len(v)} chars>" if k == "initial_prompt" and isinstance(v, str) else v)
                for k, v in whisper_kw.items()
            }
            _LOG.debug("mlx_whisper.transcribe kwargs=%r mlx_decode_meta=%r", safe_kw, mlx_decode_meta)

            t_asr0 = time.perf_counter()
            try:
                _LOG.info("mlx: starting Whisper transcribe on %s", whisper_input)
                asr_result = mlx_whisper.transcribe(whisper_input, **whisper_kw)
            finally:
                if vad_path is not None:
                    try:
                        vad_path.unlink(missing_ok=True)
                    except OSError:
                        pass

            asr_dt = time.perf_counter() - t_asr0
            n_seg0 = len(asr_result.get("segments") or [])
            _LOG.info(
                "mlx: Whisper pass done in %.2fs segments=%d language=%r",
                asr_dt,
                n_seg0,
                asr_result.get("language"),
            )

            if duration_audio <= 0.0:
                duration_audio = _audio_duration_sf(work_inp)
            asr_result = _mlx_sanitize_segments_before_tail_merge(
                asr_result, duration=duration_audio
            )
            n_seg1 = len(asr_result.get("segments") or [])
            if n_seg1 != n_seg0:
                _LOG.debug(
                    "mlx: after subtitle/end sanitize segments %d -> %d",
                    n_seg0,
                    n_seg1,
                )

            asr_result, mlx_tail_segments_merged = _mlx_merge_tail_retranscribe(
                work_inp,
                asr_result,
                tail_sec=vad_tail_sec,
                whisper_kw=whisper_kw,
            )
            asr_result, halluc_removed = _filter_hallucinated_segments(asr_result)
            n_seg2 = len(asr_result.get("segments") or [])
            _LOG.info(
                "mlx: post tail-merge + hallucination filter: final_segments=%d tail_merged=%d halluc_removed=%d",
                n_seg2,
                mlx_tail_segments_merged,
                halluc_removed,
            )

            if duration_audio <= 0.0:
                ends = [
                    float(s.get("end", 0.0)) for s in (asr_result.get("segments") or [])
                ]
                duration_audio = max(ends) if ends else 0.0

            t_diar0 = time.perf_counter()
            _LOG.info(
                "mlx: starting diarize() on full normalized wav=%s num_speakers=%r",
                work_inp,
                num_speakers,
            )
            dresult = run_diarize(
                str(work_inp),
                num_speakers=num_speakers,
                min_speakers=1,
                max_speakers=20,
            )
            diar_dt = time.perf_counter() - t_diar0
            diar = [
                DiarSeg(float(s.start), float(s.end), str(s.speaker))
                for s in dresult.segments
            ]
            _LOG.info(
                "mlx: diarize done in %.2fs raw_segments=%d audio_duration=%r",
                diar_dt,
                len(diar),
                getattr(dresult, "audio_duration", None),
            )
            if not diar:
                span = float(dresult.audio_duration or duration_audio or 1.0)
                _LOG.warning("mlx: diarize returned no segments; using single SPEAKER_00")
                diar = [DiarSeg(0.0, span, "SPEAKER_00")]

            try:
                diarize_ver = importlib.metadata.version("diarize")
            except importlib.metadata.PackageNotFoundError:
                diarize_ver = "unknown"

            asr_model_meta = _whisper_repo()
            diar_backend = "diarize"
            diar_model_meta = "diarize"
            asr_runtime_meta = "mlx (Apple Silicon)"
            diar_device_meta = "cpu"
            pipeline_meta = "mlx"
            vad_prefilter = (
                f"silero_vad ({vad_seg_count} speech segments, "
                f"speech_pad_ms={vad_pad_ms}, tail_preserve_sec={vad_tail_sec})"
            )

        enroll_dir, enroll_source = resolve_enrollment_directory()
        _LOG.info(
            "enrollment: directory=%s source=%s",
            enroll_dir,
            enroll_source,
        )
        strategy = "chronological"
        label_map: dict[str, str]
        if enroll_dir is not None:
            _LOG.debug("enrollment: running WeSpeaker embedding label map")
            emb_map = _wespeaker_embedding_label_map(diar, enroll_dir, work_inp)
            if emb_map is not None:
                label_map = emb_map
                strategy = "embedding_enrollment_wespeaker"
                _LOG.info("enrollment: strategy=%s label_map=%s", strategy, label_map)
            else:
                label_map = _chronological_label_map(diar)
                _LOG.warning(
                    "enrollment: WeSpeaker map failed; fallback chronological labels=%s",
                    label_map,
                )
        else:
            label_map = _chronological_label_map(diar)
            _LOG.info("enrollment: skipped; chronological labels=%s", label_map)

        asr_text_fix_substitutions = 0
        pieces: list[tuple[float, float, str, str]] = []
        for seg in asr_result.get("segments") or []:
            words = seg.get("words")
            if words:
                for w in words:
                    t = (w.get("word") or "").strip()
                    if not t:
                        continue
                    ws = float(w["start"])
                    we = float(w["end"])
                    lab = _speaker_for_interval(ws, we, diar, 0.5 * (ws + we))
                    name = label_map.get(lab, lab)
                    pieces.append((ws, we, t, name))
            else:
                st = float(seg["start"])
                en = float(seg["end"])
                tx = (seg.get("text") or "").strip()
                if not tx:
                    continue
                lab = _speaker_for_interval(st, en, diar, 0.5 * (st + en))
                name = label_map.get(lab, lab)
                pieces.append((st, en, tx, name))

        blocks: list[tuple[str, float, float, str]] = []
        if pieces:
            cur_name = pieces[0][3]
            cur_s, cur_e = pieces[0][0], pieces[0][1]
            parts: list[str] = [pieces[0][2]]
            for ws, we, t, name in pieces[1:]:
                if name == cur_name:
                    cur_e = we
                    parts.append(t)
                else:
                    joined = _scrub_repeated_short_token_artifacts(" ".join(parts).strip())
                    joined, nfix = apply_transcript_text_fixes(joined)
                    asr_text_fix_substitutions += nfix
                    if joined:
                        blocks.append((cur_name, cur_s, cur_e, joined))
                    cur_name, cur_s, cur_e, parts = name, ws, we, [t]
            last_joined = _scrub_repeated_short_token_artifacts(" ".join(parts).strip())
            last_joined, nfix = apply_transcript_text_fixes(last_joined)
            asr_text_fix_substitutions += nfix
            if last_joined:
                blocks.append((cur_name, cur_s, cur_e, last_joined))

        full_text = " ".join(b[3] for b in blocks)
        word_count = len(re.findall(r"\S+", full_text))
        interruptions = _count_interruptions(diar)
        proc_sec = time.perf_counter() - t_wall0

        meta = {
            "source_path": str(inp),
            "source_sha256": digest,
            "audio_duration_seconds": round(duration_audio, 3),
            "processing_time_seconds": round(proc_sec, 3),
            "word_count": word_count,
            "interruption_count": interruptions,
            "language": asr_result.get("language"),
            "pipeline": pipeline_meta,
            "asr_model": asr_model_meta,
            "diarization_backend": diar_backend,
            "diarization_model": diar_model_meta,
            "diarization_package_version": diarize_ver,
            "diarization_speaker_count_mode": num_speakers
            if num_speakers is not None
            else "auto",
            "speaker_assignment_strategy": strategy,
            "speaker_label_map": {k: label_map[k] for k in sorted(label_map.keys())},
            "diarization_device": diar_device_meta,
            "asr_runtime": asr_runtime_meta,
            "audio_input_preprocess": preprocess,
            "enrollment_source": enroll_source,
            "asr_scrub_repeat_tokens": _scrub_repeat_tokens_enabled(),
            "asr_text_fixes": asr_text_fixes_enabled(),
            "asr_text_fix_substitutions": asr_text_fix_substitutions,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if pipeline_meta == "mlx":
            meta["vad_prefilter"] = vad_prefilter
            meta["vad_speech_pad_ms"] = vad_pad_ms
            meta["vad_tail_preserve_seconds"] = vad_tail_sec
            meta["mlx_tail_segments_merged"] = mlx_tail_segments_merged
            meta["condition_on_previous_text"] = mlx_condition_on_previous_text
            meta["hallucinated_segments_removed"] = halluc_removed
            if mlx_decode_meta:
                meta["mlx_whisper_decode_overrides"] = mlx_decode_meta
        if enroll_dir is not None:
            meta["enrollment_dir"] = str(enroll_dir)

        meta_out = meta if verbose_meta else _minimal_transcript_meta(meta)
        md = _build_markdown(meta_out, blocks)
        _LOG.info(
            "writing markdown: path=%s blocks=%d words=%d processing_time=%.2fs verbose_meta=%s",
            out,
            len(blocks),
            word_count,
            proc_sec,
            verbose_meta,
        )
        _atomic_write_text(out, md)
        _LOG.info("success: wrote transcript bytes=%d", len(md.encode("utf-8")))
        return 0
    finally:
        if tmp_normalized is not None:
            try:
                tmp_normalized.unlink(missing_ok=True)
            except OSError:
                pass
        flush_run_log()
