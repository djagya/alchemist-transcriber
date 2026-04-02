"""Verify enrollment reference clips (same ffmpeg + WeSpeaker path as transcribe)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from audio_preprocess import normalize_to_pipeline_wav

_REPO_ROOT = Path(__file__).resolve().parents[1]
from enrollment_util import (
    ENROLLMENT_AUDIO_EXTENSIONS,
    resolve_enrollment_directory,
    resolve_role_clip,
)


def main() -> int:
    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv()
    p = argparse.ArgumentParser(
        description=(
            "Check enrollment clips (danil.*, therapist.*). "
            "With no --dir, uses ALCHEMIST_ENROLLMENT_DIR or ./enrollments (see README). "
            "Runs the same ffmpeg normalization as transcribe before WeSpeaker."
        )
    )
    p.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=(
            "Enrollment directory (default: from env or ./enrollments under cwd)"
        ),
    )
    args = p.parse_args()
    if args.dir is not None:
        d = args.dir.expanduser().resolve()
        source = "cli"
    else:
        d, source = resolve_enrollment_directory()
        if d is None:
            print(
                "error: no enrollment directory (set --dir, ALCHEMIST_ENROLLMENT_DIR, "
                "or create ./enrollments)",
                file=sys.stderr,
            )
            return 1
    if not d.is_dir():
        print(f"error: not a directory: {d}", file=sys.stderr)
        return 1

    roles = ("danil", "therapist")
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for role in roles:
        clip = resolve_role_clip(d, role)
        if clip is None:
            missing.append(
                f"{role}.* ({', '.join(ENROLLMENT_AUDIO_EXTENSIONS)})"
            )
        else:
            paths[role] = clip
    if missing:
        print(f"error: missing in {d}:", ", ".join(missing), file=sys.stderr)
        return 1

    print(f"enrollment_dir: {d} (source={source})")
    for role in roles:
        print(f"source: {role}: {paths[role]}")

    work_paths: dict[str, Path] = {}
    temps: list[Path] = []
    try:
        for role in roles:
            w, tmp = normalize_to_pipeline_wav(
                paths[role],
                temp_prefix=f"alchemist-enroll-check-{role}-",
                quiet=False,
            )
            work_paths[role] = w
            if tmp is not None:
                temps.append(tmp)
            print(f"pipeline_input: {role}: {w}" + (" (ffmpeg temp)" if tmp else ""))

        import numpy as np
        import wespeakerruntime as wespeaker_rt

        spk = wespeaker_rt.Speaker(lang="en")
        for role in roles:
            emb = spk.extract_embedding(str(work_paths[role]))
            v = np.asarray(emb, dtype=np.float64).reshape(-1)
            print(f"embedding {role}: dim={v.shape[0]}")
    except Exception as e:
        print(f"warning: could not load WeSpeaker ({e})", file=sys.stderr)
    finally:
        for tmp in temps:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
