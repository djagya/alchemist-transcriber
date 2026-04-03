# Alchemist transcriber

Transcribe audio with **Whisper** ASR and **speaker diarization**, writing a **Markdown** file with YAML metadata (durations, word count, interruption count, speaker map).

**Default (MLX):** **mlx-whisper** (`large-v3` MLX weights) plus **CPU** diarization via the [`diarize`](https://pypi.org/project/diarize/) package.

**Optional (PyTorch):** **openai-whisper** (default **`large-v3`** PyTorch checkpoint) plus **pyannote** [`speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1). Set `ALCHEMIST_PIPELINE=torch`, install the extra, and provide `HF_TOKEN` (accept pyannote model terms on Hugging Face).

## Requirements

- macOS with Apple Silicon (MLX ASR)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Python 3.10+
- **[ffmpeg](https://ffmpeg.org/) on your PATH** (strongly recommended). The pipeline normalizes input to **16 kHz mono 16-bit PCM WAV** before ASR and diarization so containers like Voice Memos **`.qta`**, odd **`.m4a`**, etc. decode reliably. Without ffmpeg, the original file is used and you may see warnings or failures from Python decoders.

### Why WAV @ 16 kHz mono (not m4a)?

- **16 kHz** matches what Whisper was trained on; resampling once in ffmpeg avoids duplicate work and keeps timestamps consistent.
- **PCM WAV** is read consistently by **soundfile** (diarization, duration, enrollment crops) and avoids torchaudio/soundfile gaps on some AAC/MP4 variants.
- **m4a/AAC** as an intermediate would mean another lossy encode/decode step and still depends on codec support in downstream libraries.

Normalization is usually **much faster than real-time**; the temp WAV is deleted after the run.

**Enrollment** reference clips (`danil.*`, `therapist.*`) go through the **same** ffmpeg step before WeSpeaker extracts embeddings, so `.m4a` and other formats behave like the main session file.

## Install

```bash
cd alchemist-transcriber
uv sync
```

This creates `.venv` and installs the `alchemist-transcriber` package in editable mode.

Optional: copy **`.env.example`** to **`.env`** in the directory you run `uv run transcribe` from and set secrets (e.g. `HF_TOKEN` for the torch pipeline). **`.env`** is gitignored. The CLI loads it automatically via [python-dotenv](https://pypi.org/project/python-dotenv/) (existing shell variables take precedence).

### PyTorch + pyannote pipeline (optional)

```bash
uv sync --extra torch-pipeline
```

1. Create a Hugging Face access token and set **`HF_TOKEN`** (or **`HUGGING_FACE_HUB_TOKEN`**).
2. Accept the conditions on the Hub for **all** of these (same account as `HF_TOKEN`): **[speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)**, **[segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)**, **[wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)**.
3. Run with:

```bash
export ALCHEMIST_PIPELINE=torch
export HF_TOKEN=...   # your token
uv run transcribe --input /path/to/session.m4a --output /path/to/out-torch.md
```

Use **`ALCHEMIST_OPENAI_WHISPER_MODEL`** for another Whisper size (see `whisper.available_models()`). **`ALCHEMIST_WHISPER_DEVICE`** is `auto` (CUDA → MPS → CPU), or force `cpu` / `mps` / `cuda`. **`ALCHEMIST_DIARIZATION_DEVICE`** defaults to **`cpu`** for pyannote (set `mps` or `cuda` if you want diarization on GPU).

## Transcribe (default speaker naming)

Diarization clusters are named by **time order**: the first speaker in the file → **Danil**, the second → **Therapist** (then `Speaker 3`, …). The default MLX pipeline does **not** require a Hugging Face token for diarization; the torch/pyannote path **does** (see above).

If a directory named **`enrollments`** exists in the **current working directory** (where you run the command), it is used automatically for **speaker enrollment** when it contains usable `danil.*` and `therapist.*` clips (see below). Override with `ALCHEMIST_ENROLLMENT_DIR`; set it to `none` to ignore `./enrollments`.

```bash
uv run transcribe --input /path/to/session.m4a --output /path/to/out.md
```

Use **`--output`** with any **absolute or relative path**; parent directories are created if missing.

**YAML frontmatter:** by default the file gets a **short** header (source hash, duration, word count, language, pipeline, models, speaker map, timestamp). Use **`--verbose`** (or **`ALCHEMIST_VERBOSE_META=1`**) to include the **full** debug block (VAD segment counts, tail merge, package versions, `enrollment_dir`, scrub/fix flags, MLX decode overrides, etc.).

**Stderr:** **`--quiet`** or **`ALCHEMIST_QUIET=1`** suppresses ffmpeg normalization warnings only.

```bash
uv run transcribe --input /path/to/session.m4a --output /path/to/out.md
uv run transcribe --input /path/to/session.m4a --output /path/to/out.md --verbose
uv run transcribe --input /path/to/session.m4a --output /path/to/out.md --quiet
```

### Idempotency

If `out.md` already exists and its YAML frontmatter `source_sha256` matches the current input file, the command **exits immediately** without re-running models.

## Speaker enrollment (optional)

Enrollment gives **reference clips** of each named speaker so anonymous diarization labels (`SPEAKER_00`, `SPEAKER_01`) are mapped to **Danil** and **Therapist** by **voice similarity** (WeSpeaker embeddings), instead of “whoever spoke first.”

### Automatic `./enrollments`

From the repo root (or any directory where you run `transcribe`), create:

```text
enrollments/
  danil.<ext>
  therapist.<ext>
```

Supported extensions (first match wins): **`.wav`**, **`.m4a`**, **`.mp3`**, **`.flac`**, **`.ogg`**. Stems are matched case-insensitively (`Danil.m4a` is fine).

Then run `uv run transcribe ...` **without** setting any env var; the tool picks up `./enrollments` if that folder exists.

### 1. Prepare clips

| Clip | Content |
|------|--------|
| `danil.*` | Several seconds of **only Danil** speaking (clear solo speech). |
| `therapist.*` | Several seconds of **only the therapist** speaking. |

Tips:

- **5–30 seconds** each is usually enough; clarity beats length.
- Prefer the **same recording conditions** as the session (same mic/room/codec family) when possible.

You can cut clips with **ffmpeg**, for example:

```bash
mkdir -p enrollments
ffmpeg -i session.m4a -ss 00:01:10 -t 12 -ac 1 -ar 16000 -c:a pcm_s16le enrollments/danil.wav
ffmpeg -i session.m4a -ss 00:05:00 -t 12 -ac 1 -ar 16000 -c:a pcm_s16le enrollments/therapist.wav
```

Adjust `-ss` (start) and `-t` (duration) to solo stretches. You can also copy **`.m4a`** (or other supported) files into `enrollments/` as `danil.m4a` / `therapist.m4a` if WeSpeaker accepts them (same as the main transcriber pipeline).

### 2. Verify files (optional)

From the same working directory where `./enrollments` lives:

```bash
uv run alchemist-check-enrollment
```

Or point at a specific folder:

```bash
uv run alchemist-check-enrollment --dir /absolute/path/to/enrollments
```

### 3. Override or disable

Use **`ALCHEMIST_ENROLLMENT_DIR`** to force a different directory. Set it to **`none`**, **`off`**, **`false`**, or **`0`** to **skip** enrollment even if `./enrollments` exists:

```bash
export ALCHEMIST_ENROLLMENT_DIR=none
uv run transcribe --input /path/to/session.m4a --output /path/to/out.md
```

If embedding enrollment fails (missing clips, unreadable audio, etc.), the tool **falls back** to chronological naming. The default frontmatter includes **`speaker_assignment_strategy`** (`embedding_enrollment_wespeaker` vs `chronological`). **`enrollment_source`** and **`enrollment_dir`** appear only with **`--verbose`** / **`ALCHEMIST_VERBOSE_META`**.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `ALCHEMIST_PIPELINE` | `mlx` (default) or `torch` (openai-whisper + pyannote; requires `uv sync --extra torch-pipeline` and `HF_TOKEN`). |
| `ALCHEMIST_ENROLLMENT_DIR` | Optional. If set, use this directory for clips. If unset, use `./enrollments` when present. `none` / `off` / `false` / `0` disables enrollment. |
| `ALCHEMIST_NUM_SPEAKERS` | Default `2`. Set to `auto` for automatic speaker count. |
| `ALCHEMIST_LANGUAGE` | Whisper language code (e.g. `ru`, `en`). **If the variable is unset, default is `ru`.** Set to `auto` or an empty value to auto-detect. |
| `ALCHEMIST_INITIAL_PROMPT` | Overrides the built-in default `initial_prompt` (Russian therapy: relationships, substances/addiction, life themes, sample names). Omit to use the default. |
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | **Required** for `ALCHEMIST_PIPELINE=torch` (pyannote). Optional for MLX if Hub access is needed. |
| `ALCHEMIST_SKIP_FFMPEG_NORMALIZE` | If `1` / `true` / `yes`, skip ffmpeg and pass the original file straight through (debug only). |
| `ALCHEMIST_WHISPER_REPO` | MLX pipeline only: MLX Whisper weights on the Hub (default `mlx-community/whisper-large-v3-mlx`). Turbo: `mlx-community/whisper-large-v3-turbo`. |
| `ALCHEMIST_VAD_SPEECH_PAD_MS` | MLX only: Silero `speech_pad_ms` when building the VAD-masked WAV (default `200`). |
| `ALCHEMIST_VAD_TAIL_PRESERVE_SEC` | MLX only: last *N* seconds of audio are **not** zeroed by VAD so quiet lines after goodbye stay audible to Whisper (default `25`; set `0` to disable). |
| `ALCHEMIST_MLX_TAIL_RETRANSCRIBE` | MLX only: `1` (default) runs a second Whisper pass on that same tail window (from the **normalized** file) and merges segments missing from the full pass; `0` disables. |
| `ALCHEMIST_MLX_CONDITION_ON_PREVIOUS_TEXT` | MLX only: set to `1` to enable Whisper `condition_on_previous_text` (carry text into the next window; may help names/context; can increase repetition hallucinations and runtime). Default **off** (omit or `0`). |
| `ALCHEMIST_SCRUB_REPEAT_TOKENS` | `1` (default): remove runs of repeated single-letter tokens (common silence hallucinations like «в в в») when building each speaker block; `0` disables. |
| `ALCHEMIST_ASR_TEXT_FIXES` | `1` (default): after scrub, normalize Whisper hyphen spacing (e.g. `что -то` → `что-то`, `из - за` → `из-за`) and apply small name/typo rules from `src/transcript_fixes.py`; `0` disables. Frontmatter: `asr_text_fixes`, `asr_text_fix_substitutions`. |
| `ALCHEMIST_QUIET` | `1` / `true` / `yes`: same as **`--quiet`** — suppress ffmpeg warnings on stderr only. |
| `ALCHEMIST_VERBOSE_META` | `1` / `true` / `yes`: same as **`--verbose`** — **full** YAML frontmatter (default is short). |
| `ALCHEMIST_MLX_NO_SPEECH_THRESHOLD` | Optional. MLX only: float passed to `mlx_whisper.transcribe` (library default `0.6`). |
| `ALCHEMIST_MLX_LOGPROB_THRESHOLD` | Optional. MLX only (default in lib: `-1.0`). |
| `ALCHEMIST_MLX_COMPRESSION_RATIO_THRESHOLD` | Optional. MLX only (default in lib: `2.4`). |
| `ALCHEMIST_MLX_HALLUCINATION_SILENCE_THRESHOLD` | Optional. MLX only; unset in lib unless you set it. |
| `ALCHEMIST_OPENAI_WHISPER_MODEL` | Torch pipeline only: openai-whisper model name (default `large-v3`). |
| `ALCHEMIST_WHISPER_DEVICE` | Torch pipeline only: `auto`, `cpu`, `mps`, or `cuda`. |
| `ALCHEMIST_DIARIZATION_DEVICE` | Torch pipeline only: `cpu` (default), `mps`, or `cuda`. |

With **`--verbose`** / **`ALCHEMIST_VERBOSE_META`**, the frontmatter also includes fields such as **`audio_input_preprocess`** (`ffmpeg_pcm_s16le_16000hz_mono_wav` when normalization ran, or `none` when skipped or ffmpeg was unavailable), **`enrollment_source`**, MLX VAD/tail counters, and package versions.

## Project layout

- `.env.example` — environment variable template (copy to `.env`; `.env` is gitignored)
- `src/cli.py` — main transcription pipeline
- `src/torch_pipeline.py` — optional PyTorch Whisper + pyannote path
- `src/diar_types.py` — shared diarization segment type
- `src/audio_preprocess.py` — shared ffmpeg → 16 kHz mono WAV (session + enrollment)
- `src/enrollment_util.py` — resolves `./enrollments` and clip paths
- `src/check_enrollment.py` — enrollment file check helper
- `SKILL.md` — short notes for OpenClaw-style agents

## License

Add a license if you publish the repo; dependencies have their own licenses (see PyPI pages for `mlx-whisper`, `diarize`, etc.).
