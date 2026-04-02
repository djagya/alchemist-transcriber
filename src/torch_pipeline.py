"""OpenAI Whisper (PyTorch) + pyannote.audio diarization."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio
from pyannote.audio import Pipeline

from diar_types import DiarSeg

PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

_hf_hub_pyannote_patched = False


def _patch_hf_hub_for_pyannote() -> None:
    """pyannote.audio 3.x passes use_auth_token= to hf_hub_download; huggingface_hub 1.x expects token=."""
    global _hf_hub_pyannote_patched
    if _hf_hub_pyannote_patched:
        return
    import huggingface_hub
    import pyannote.audio.core.model as model_mod
    import pyannote.audio.core.pipeline as pl_mod

    _real = huggingface_hub.hf_hub_download

    def _compat(*args, **kwargs):
        if "use_auth_token" in kwargs:
            uat = kwargs.pop("use_auth_token")
            if kwargs.get("token") is None:
                kwargs["token"] = uat
        return _real(*args, **kwargs)

    huggingface_hub.hf_hub_download = _compat
    pl_mod.hf_hub_download = _compat
    model_mod.hf_hub_download = _compat
    _hf_hub_pyannote_patched = True


def _annotation_to_segments(annotation: Any) -> list[DiarSeg]:
    out: list[DiarSeg] = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        out.append(DiarSeg(float(segment.start), float(segment.end), str(label)))
    out.sort(key=lambda x: (x.start, x.end))
    return out


def _whisper_torch_device() -> str:
    raw = (os.environ.get("ALCHEMIST_WHISPER_DEVICE") or "auto").strip().lower()
    if raw in ("cuda", "mps", "cpu"):
        return raw
    if torch.cuda.is_available():
        return "cuda"
    # openai-whisper hits unsupported sparse MPS ops on Apple Silicon; CPU is reliable.
    if sys.platform == "darwin":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pyannote_torch_device() -> torch.device:
    raw = (os.environ.get("ALCHEMIST_DIARIZATION_DEVICE") or "cpu").strip().lower()
    if raw == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if raw == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_openai_whisper_pyannote(
    work_audio: Path,
    *,
    hf_token: str,
    initial_prompt: str | None,
    language: str | None,
    whisper_model: str,
    num_speakers: int | None,
) -> tuple[dict[str, Any], list[DiarSeg], float, str, str]:
    """Transcribe with openai-whisper, diarize with pyannote 3.1.

    Returns:
        whisper_result, diar_segments, duration_seconds, whisper_device, diar_device_str
    """
    import whisper

    wdev = _whisper_torch_device()
    model = whisper.load_model(whisper_model, device=wdev)

    transcribe_kw: dict[str, Any] = {
        "word_timestamps": True,
        "verbose": False,
    }
    if initial_prompt:
        transcribe_kw["initial_prompt"] = initial_prompt
    if language:
        transcribe_kw["language"] = language

    result = model.transcribe(str(work_audio), **transcribe_kw)
    del model
    if wdev == "cuda":
        torch.cuda.empty_cache()

    segs = result.get("segments") or []
    duration_audio = 0.0
    if segs:
        duration_audio = float(segs[-1].get("end", 0.0))
    if duration_audio <= 0.0:
        try:
            info = torchaudio.info(str(work_audio))
            duration_audio = info.num_frames / float(info.sample_rate)
        except Exception:
            pass

    waveform, sample_rate = torchaudio.load(str(work_audio))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    dia_dev = _pyannote_torch_device()
    _patch_hf_hub_for_pyannote()
    pipeline = Pipeline.from_pretrained(PYANNOTE_MODEL, use_auth_token=hf_token)
    if pipeline is None:
        raise RuntimeError(
            f"Could not load {PYANNOTE_MODEL} from Hugging Face (gated or auth). "
            "Use a valid HF_TOKEN and accept terms for speaker-diarization-3.1, "
            "segmentation-3.0, and wespeaker-voxceleb-resnet34-LM (see README), then retry."
        )
    pipeline.to(dia_dev)

    audio_file: dict[str, Any] = {
        "waveform": waveform,
        "sample_rate": int(sample_rate),
    }
    diar_kw: dict[str, Any] = {}
    if num_speakers is not None:
        diar_kw["num_speakers"] = num_speakers

    diarization = pipeline(audio_file, **diar_kw)

    diar = _annotation_to_segments(diarization)
    return result, diar, duration_audio, wdev, str(dia_dev)
