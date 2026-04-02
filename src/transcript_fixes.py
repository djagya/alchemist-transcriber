"""Deterministic post-ASR text fixes: hyphen spacing (Whisper «что -то») and name/typo rules."""

from __future__ import annotations

import os
import re
from typing import Iterable

# Word chars for Russian + Latin (Whisper output).
_W = r"0-9A-Za-zА-Яа-яЁё"
# Particles that form compounds with a preceding word: что-то, как-нибудь, все-таки, ну-ка, …
_HYPHEN_PARTICLE_CAP = r"(то|нибудь|либо|таки|ка)\b"

# Longer multi-word compounds first (before generic word + particle).
# Whisper often drops the space after the hyphen: «какую -то», «во -первых».
_FIXED_PHRASES: list[tuple[str, str]] = [
    (rf"(?i)([{_W}]+)\s+-\s*за\b", r"\1-за"),  # из -за → из-за
    (rf"(?i)([{_W}]+)\s+-\s*под\b", r"\1-под"),
    (r"(?i)по\s+-\s*крайней\s+мере\b", "по-крайней мере"),
    (r"(?i)во\s+-\s*первых\b", "во-первых"),
    (r"(?i)во\s+-\s*вторых\b", "во-вторых"),
    (r"(?i)во\s+-\s*третьих\b", "во-третьих"),
]

# Ordered: longer / more specific patterns first.
_NAME_AND_TYPO: list[tuple[re.Pattern[str], str]] = [
    # ASR drops «рас-» in «расстались» after name Тея (therapy corpus).
    (
        re.compile(r"\bс\s+Тейер\s+остались\b", re.UNICODE | re.IGNORECASE),
        "с Теей расстались",
    ),
    (re.compile(r"\bс\s+Тейер\b", re.UNICODE), "с Теей"),
    (re.compile(r"\bс\s+тейер\b", re.UNICODE), "с Теей"),
    (re.compile(r"\bТейер\b", re.UNICODE), "Тея"),
    (re.compile(r"\bтейер\b", re.UNICODE), "Тея"),
    (re.compile(r"\bбуквительно\b", re.UNICODE), "буквально"),
    (re.compile(r"\bстранясь\b", re.UNICODE), "сторонясь"),
    # ASR: «трапез» for «терапевт» before «хочет/говорит/сидит» (therapy context).
    (re.compile(r"\bтрапез\b(?=\s+(?:хочет|говорит|сидит)\b)", re.UNICODE | re.IGNORECASE), "терапевт"),
]

_HYPHEN_GENERIC = re.compile(
    rf"(?i)([{_W}]+)\s+-\s*{_HYPHEN_PARTICLE_CAP}", re.UNICODE
)


def asr_text_fixes_enabled() -> bool:
    raw = os.environ.get("ALCHEMIST_ASR_TEXT_FIXES", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _collapse_spaces(s: str) -> str:
    return re.sub(r" {2,}", " ", s).strip()


def _apply_substitution_patterns(
    text: str, patterns: Iterable[tuple[re.Pattern[str], str]]
) -> tuple[str, int]:
    n = 0
    for rx, repl in patterns:
        text, k = rx.subn(repl, text)
        n += k
    return text, n


def _normalize_hyphen_spacing(text: str) -> tuple[str, int]:
    n = 0
    for pat, repl in _FIXED_PHRASES:
        rx = re.compile(pat, re.UNICODE)
        text, k = rx.subn(repl, text)
        n += k
    text, k = _HYPHEN_GENERIC.subn(r"\1-\2", text)
    n += k
    return text, n


def apply_transcript_text_fixes(text: str) -> tuple[str, int]:
    """Return (fixed_text, number_of_substitutions). No-op if disabled or empty."""
    if not text or not asr_text_fixes_enabled():
        return text, 0
    total = 0
    t, k = _normalize_hyphen_spacing(text)
    total += k
    t, k = _apply_substitution_patterns(t, _NAME_AND_TYPO)
    total += k
    return _collapse_spaces(t), total
