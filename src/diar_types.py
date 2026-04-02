"""Shared diarization segment type for MLX and PyTorch pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiarSeg:
    start: float
    end: float
    label: str
