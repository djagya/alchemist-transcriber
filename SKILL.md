# Alchemist transcriber (OpenClaw)

Use this repo to turn **therapy session audio** into a **markdown transcript**: diarized speakers, timestamps per turn, and a **YAML header** for automation. Intended when the user (or OpenClaw) supplies a file path and needs a stable `.md` artifact.

## Best quality (recommended)

For **Russian therapy** recordings on **Apple Silicon**, use the **defaults** — no need to set `ALCHEMIST_PIPELINE`:

| Stage | What runs |
|-------|-----------|
| **Normalize** | **ffmpeg** → 16 kHz mono PCM WAV (reliable decode for `.m4a`, `.qta`, etc.) |
| **VAD prefilter (MLX only)** | **Silero** masks non-speech before Whisper; **tail preserve** (last ~25 s) keeps quiet outros |
| **ASR** | **mlx-whisper** with Hub weights (default `mlx-community/whisper-large-v3-mlx`) |
| **Diarization** | **[`diarize`](https://pypi.org/project/diarize/)** on the **full normalized** WAV (Silero + WeSpeaker ONNX on CPU) |
| **Labels** | Optional **`./enrollments`** with `danil.*` / `therapist.*` → **WeSpeaker** mapping to names; else **chronological** Danil / Therapist |
| **Cleanup** | Repeat-token scrub + **`transcript_fixes`** (hyphens, names like Тея, therapy phrases) |

**Do not** switch to `torch` unless you have a reason (pyannote needs **`HF_TOKEN`** and Hub model acceptance). The MLX + `diarize` path is the supported “good transcript” stack here.

**Requirements:** **ffmpeg** on `PATH`, **uv**, macOS + Apple Silicon for MLX. Install: `cd …/alchemist-transcriber && uv sync`.

## Commands OpenClaw should run

Always pass **absolute paths** for `--input` and **`--output`** so the markdown lands where the orchestrator expects (any folder; parents are created).

```bash
cd /path/to/alchemist-transcriber
uv run transcribe \
  --input "/absolute/path/to/session.m4a" \
  --output "/absolute/path/to/transcripts/session.md"
```

- **`--input`**: one audio file (session).
- **`--output`**: target **`.md`** path (required). Idempotency uses `source_sha256` in the file header vs the input file.
- **YAML `---` block:** **short by default** (hash, duration, word count, language, pipeline, ASR/diarization ids, speaker map, timestamp). For **debug / tuning**, add **`--verbose`** or **`ALCHEMIST_VERBOSE_META=1`** to dump the full header (VAD counts, tail merge, versions, `enrollment_dir`, scrub flags, MLX overrides, etc.).
- **`--quiet`** or **`ALCHEMIST_QUIET=1`**: only suppresses **ffmpeg** warnings on stderr; it does **not** change the frontmatter (default is already short).

**Working directory:** Run from the repo root (or anywhere **`uv`** resolves the project). If **`./enrollments`** exists in the **current working directory** and contains `danil.*` + `therapist.*`, enrollment runs automatically. To force clips from elsewhere: `export ALCHEMIST_ENROLLMENT_DIR=/absolute/path/to/enrollments`. To disable: `export ALCHEMIST_ENROLLMENT_DIR=none`.

## When to use

- User attaches or paths an audio file and wants a **structured transcript** (two speakers, therapy-oriented defaults).
- OpenClaw post-processes sessions: run `transcribe`, then read the `## Transcript` body and/or YAML.

## Setup (once per machine)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.10+.
2. `cd alchemist-transcriber && uv sync`.
3. Ensure **ffmpeg** is installed and on `PATH`.
4. Optional: copy **`.env.example`** → **`.env`** (e.g. `ALCHEMIST_QUIET=1` for silent stderr in batch runs).
5. Optional enrollment: see **README.md** (`enrollments/danil.*`, `therapist.*`, `uv run alchemist-check-enrollment`).

## Behaviour to rely on

- **Idempotent:** if `--output` exists and frontmatter **`source_sha256`** matches the current input, exit **0** without re-running models.
- **Non-interactive:** no prompts.
- **Speaker names:** without enrollment, first two clusters in time order → **Danil**, **Therapist**; with enrollment → **WeSpeaker** match when clips are valid.
- **Language:** default **`ru`** if `ALCHEMIST_LANGUAGE` is unset; use `auto` only if the session language is mixed or unknown.

## Useful environment variables (summary)

| Variable | Role |
|----------|------|
| `ALCHEMIST_VERBOSE_META` | Full YAML frontmatter (same as **`--verbose`**) |
| `ALCHEMIST_QUIET` | Suppress ffmpeg stderr only (same as **`--quiet`**) |
| `ALCHEMIST_PIPELINE` | `mlx` (default) or `torch` (only if pyannote path is required) |
| `ALCHEMIST_INITIAL_PROMPT` | Override Whisper hint; default is Russian therapy-oriented (themes + names) |
| `ALCHEMIST_LANGUAGE` | `ru` default; `auto` or empty for detect |
| `ALCHEMIST_NUM_SPEAKERS` | Default `2`; `auto` for automatic count |
| `ALCHEMIST_ENROLLMENT_DIR` | Override or disable (`none`) enrollment directory |
| `ALCHEMIST_WHISPER_REPO` | MLX Hub repo (e.g. turbo variant) |
| `ALCHEMIST_VAD_*` / `ALCHEMIST_MLX_TAIL_RETRANSCRIBE` | Tune VAD and tail re-transcribe (defaults are tuned for quiet endings) |
| `ALCHEMIST_ASR_TEXT_FIXES` | `0` to disable `src/transcript_fixes.py` rules |

Full table: **README.md**.

## Output shape

- **`---` YAML `---`** then **`## Transcript`** and blocks: `**Name** (MM:SS.mmm–MM:SS.mmm)` + paragraph text.
- **Default** frontmatter: `source_path`, `source_sha256`, `audio_duration_seconds`, `word_count`, `interruption_count`, `language`, `pipeline`, `asr_model`, `diarization_backend`, `speaker_assignment_strategy`, `speaker_label_map`, `generated_at_utc`.
- **`--verbose`:** adds debug keys (VAD, tail, versions, enrollment path, scrub/fix toggles, MLX overrides, etc.).

See **README.md** for torch/HF token details and enrollment clips.
