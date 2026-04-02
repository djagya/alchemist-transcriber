# Alchemist transcriber (OpenClaw)

Use this repo to turn user-supplied audio into a **markdown transcript** with **diarized speakers** (default labels **Danil** and **Therapist**) and **YAML metadata** (durations, word count, interruption count).

**Default:** **mlx-whisper** (`large-v3` MLX weights) + **CPU diarization** via [`diarize`](https://pypi.org/project/diarize/) (Silero VAD + WeSpeaker ONNX).

**Alternate:** `ALCHEMIST_PIPELINE=torch` — **openai-whisper** (`large-v3` by default) + **pyannote** `speaker-diarization-3.1`. Requires `uv sync --extra torch-pipeline`, **`HF_TOKEN`**, and Hub acceptance for pyannote models.

## When to use

- The user attaches or points to an audio file and wants a clean `.md` transcript with speaker turns and quality-oriented ASR.

## One-shot command

From the repo root (after setup below):

```bash
uv run transcribe --input "/absolute/path/to/audio.m4a" --output "/absolute/path/to/out.md"
```

## Setup (once per machine)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and use Python 3.10+ (Apple Silicon Mac: M4 is supported; ASR runs via **MLX**; diarization runs on **CPU** and is chosen for speed there).
2. Install dependencies into the project environment:

   ```bash
   cd /path/to/alchemist-transcriber
   uv sync
   ```

   For the PyTorch + pyannote path: `uv sync --extra torch-pipeline` and set **`HF_TOKEN`** (see README). Optional: copy **`.env.example`** → **`.env`** in the run directory; the CLIs load it automatically.

   Run the CLI with `uv run transcribe ...`. With an activated venv: `transcribe ...`. See **README.md** for enrollment (`ALCHEMIST_ENROLLMENT_DIR`, `danil.wav` / `therapist.wav`) and `uv run alchemist-check-enrollment --dir ...`.

3. **ffmpeg** on PATH is expected: the **session file** and **enrollment** clips (`danil.*`, `therapist.*`) are normalized to **16 kHz mono PCM WAV** before ASR/diarization and before WeSpeaker embeddings. Without ffmpeg, originals are used and the tool may warn.
4. **Hugging Face**: default MLX diarization needs no token. **`ALCHEMIST_PIPELINE=torch`** requires **`HF_TOKEN`** for pyannote. MLX Whisper Hub limits: set `HF_TOKEN` as usual.

## Behaviour the agent should rely on

- **Idempotent**: if `--output` already exists and its YAML frontmatter `source_sha256` matches the current input file, the script **exits 0 immediately** (no rework).
- **No prompts**: non-interactive only.
- **Speaker names**: without enrollment, the **first two** diarization clusters in **chronological order** are named **Danil** then **Therapist**; further clusters become `Speaker 3`, etc.
- **Optional enrollment**: if **`./enrollments`** exists in the **cwd**, it is used automatically for **`danil.*` / `therapist.*`** clips (e.g. `.wav`, `.m4a`). Override with `ALCHEMIST_ENROLLMENT_DIR`; set to `none` to ignore `./enrollments`. Mapping uses **WeSpeaker** embeddings; on failure it falls back to chronological naming.

## Useful environment variables

| Variable | Role |
|----------|------|
| `ALCHEMIST_PIPELINE` | `mlx` (default) or `torch` (openai-whisper + pyannote) |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | **Required** for `torch` pipeline; optional for MLX Hub edge cases |
| `ALCHEMIST_INITIAL_PROMPT` | Optional override; default is Russian therapy-oriented (relationships, substances/addiction, life themes, names e.g. Тея, Лара) |
| `ALCHEMIST_LANGUAGE` | Default **`ru`** when env var absent; set `auto` or empty for auto-detect; or `en`, etc. |
| `ALCHEMIST_NUM_SPEAKERS` | Default `2` (therapy-style); set to `auto` for automatic speaker count |
| `ALCHEMIST_ENROLLMENT_DIR` | Optional override path for clips; unset uses `./enrollments` if present; `none`/`off`/`false`/`0` disables |
| `ALCHEMIST_SKIP_FFMPEG_NORMALIZE` | `1`/`true`/`yes` skips ffmpeg (original file only) |
| `ALCHEMIST_WHISPER_REPO` | MLX pipeline: Hub repo for MLX Whisper (default large-v3; turbo: `mlx-community/whisper-large-v3-turbo`) |
| `ALCHEMIST_VAD_SPEECH_PAD_MS` / `ALCHEMIST_VAD_TAIL_PRESERVE_SEC` | MLX VAD: Silero `speech_pad_ms` (default 200); **tail preserve** (default 25 s) copies raw audio at EOF into the VAD-masked WAV |
| `ALCHEMIST_MLX_TAIL_RETRANSCRIBE` | Default on: second Whisper pass on that tail (normalized WAV) + merge, fixes dropped quiet closing lines after full pass |
| `ALCHEMIST_MLX_CONDITION_ON_PREVIOUS_TEXT` | Default off; set `1` for mlx_whisper `condition_on_previous_text` (context carry-over; may help names, may worsen repeat hallucinations) |
| `ALCHEMIST_SCRUB_REPEAT_TOKENS` | Default on: strip repeated 1-letter token runs (e.g. «в в в») from merged transcript blocks |
| `ALCHEMIST_ASR_TEXT_FIXES` | Default on: hyphen spacing + name/typo rules in `src/transcript_fixes.py`; `0` disables |
| `ALCHEMIST_MLX_*` decode thresholds | Optional MLX-only floats: `ALCHEMIST_MLX_NO_SPEECH_THRESHOLD`, `ALCHEMIST_MLX_LOGPROB_THRESHOLD`, `ALCHEMIST_MLX_COMPRESSION_RATIO_THRESHOLD`, `ALCHEMIST_MLX_HALLUCINATION_SILENCE_THRESHOLD` → `mlx_whisper.transcribe` |
| `ALCHEMIST_OPENAI_WHISPER_MODEL` | Torch pipeline: openai-whisper model (default `large-v3`) |
| `ALCHEMIST_WHISPER_DEVICE` / `ALCHEMIST_DIARIZATION_DEVICE` | Torch only: ASR / pyannote device (`auto`/`cpu`/`mps`/`cuda`; diarization defaults to `cpu`) |

## Output

Markdown with a YAML frontmatter block plus a `## Transcript` section: blocks labeled by speaker with time ranges. Frontmatter includes `audio_duration_seconds`, `processing_time_seconds`, `word_count`, `interruption_count`, `audio_input_preprocess`, model ids, `diarization_backend`, and `source_sha256` for idempotency.
