# Egocentric Video Auto-Labeler

Automated **subtask annotation** of egocentric (first-person) video against a
customisable Standard Operating Procedure (SOP). The output is one continuous,
time-ordered timeline of subtasks for the whole video:

```
[0.000-3.500] Pick the screwdriver from the toolbox with the right hand.
[3.500-7.000] Walk to the workbench.
[7.000-11.200] Tighten the screw on the panel with the right hand.
```

The pipeline is three stages:

1. **Global-context stage** — sample a small set of coarse frames across the
   entire video, get a brief description of each as a single-image call
   (works even on single-image local servers like Ollama), and assemble a
   whole-video timeline used to keep object naming and timing consistent.
2. **Chunked subtask segmentation** (default) — sample frames at the requested
   rate (e.g. `--target-fps 2`), split them into overlapping chunks, and ask the
   SOP prompt for a subtask timeline per chunk. The global timeline **and** a
   rolling **prior-chunks summary** (subtasks already annotated in earlier
   chunks of the same video) are injected into every chunk prompt, so the
   annotator stays consistent across chunk boundaries.
3. **Stitch + de-duplicate** — parse each chunk's `[start-end] sentence.` lines
   and merge them (removing duplicates introduced by chunk overlap) into one
   continuous whole-video timeline.

Everything for the whole video is written into a single `.txt` file (plus
optional CSV/JSON for downstream tooling). Runs on a **local vision-language
model** by default, or against any **API** (OpenAI, Anthropic, or an
OpenAI-compatible endpoint like Ollama / vLLM / LM Studio / Together / Groq).

## Features

- **Subtask-segmentation by default**: chunked processing stitched into one continuous, de-duplicated whole-video timeline.
- **Backend choice**: local HuggingFace VLM, cloud/local API, *or* a bundled HTTP GPU worker you run on a remote GPU box (storage and outputs stay on the client machine).
- **Global-context timeline** built from coarse whole-video sampling, compatible with single-image local servers.
- **Configurable frame sampling**: rate-based `--target-fps` (default 2 fps) OR fixed `--num-frames`, with optional time window.
- **Tunable chunking**: `--chunk-frames` / `--chunk-overlap` trade context vs cost, with automatic boundary de-duplication.
- **Strict, machine-parsable SOP output**: `[start-end] instructional sentence.`
- **Fully editable SOP prompt** at `prompts/default_sop.txt`.
- **Alternate modes** still available: `context-window`, `per-frame`, `multi-frame`.
- **Broad video format support** via decoder chain: `decord` → `pyav` → `opencv` → `imageio` (mp4, mov, avi, mkv, webm, flv, m4v, wmv, mpg, mpeg, 3gp, ts, mts, m2ts, ogv, ...).
- **Batch mode**: point `--input` at a directory to label every video found.
- **Robust**: retries for API calls, decoder fallbacks, per-video error isolation.
- **Text output** with metadata header + optional JSON sidecar and CSV timeline.

## Recommended default local model

**Default (tuned for 6 GB VRAM laptops, e.g. RTX 4050 Mobile):**
`Qwen/Qwen2.5-VL-3B-Instruct` loaded in **4-bit** — Qwen-quality multi-frame
egocentric understanding at ~3 GB VRAM, leaving room for frames and KV cache.

If you have more VRAM, switch to the 7B variant with `--model Qwen/Qwen2.5-VL-7B-Instruct`.

| Model                                | Approx VRAM | Notes                                       |
| ------------------------------------ | ----------- | ------------------------------------------- |
| **`Qwen/Qwen2.5-VL-3B-Instruct` 4-bit** | **~3 GB**  | **Default.** Fits 6 GB laptops.             |
| `Qwen/Qwen2.5-VL-3B-Instruct` bf16   | ~6 GB       | Tight on a 4050; often OOMs with 8+ frames. |
| `Qwen/Qwen2.5-VL-7B-Instruct` 4-bit  | ~5.5–6 GB   | Borderline on 6 GB.                         |
| `Qwen/Qwen2.5-VL-7B-Instruct` bf16   | ~16 GB      | Best quality if you have the VRAM.          |
| `openbmb/MiniCPM-V-2_6` 4-bit        | ~5.5 GB     | Alternative architecture.                   |

Switch with `--model <hf_id>` and toggle 4-bit with `--load-in-4bit`.

## Install

```bash
# 1. Create a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate

# 2. Install PyTorch first, matched to your platform:
#    https://pytorch.org/get-started/locally/
#    e.g. CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest.
pip install -r requirements.txt

# 4. (Optional) copy the env template for API keys.
cp .env.example .env
```

`ffmpeg` should be available on your PATH for the widest codec coverage
(`imageio-ffmpeg` ships a static binary as a fallback).

## Quick start

```bash
# Sensible default: whole-video subtask timeline using the local Qwen2.5-VL-3B
# in 4-bit, sampled at 2 fps, chunked with rolling prior-summary context.
python main.py --input path/to/clip.mp4
```

That single command sits at a reasonable operating point for a 6 GB laptop GPU.
Everything else below is optional tuning for different backends, hardware,
video lengths, and output formats.

## Ways to run — full catalog

Every knob shown here can also be set in `config.yaml`. CLI flags win over the
config file. Any of these can be combined.

> Reference: run `python main.py --help` to see every flag and its default at
> once. This section groups them by decision.

### 1. Pick a backend

The tool talks to exactly one backend per run: either a **local** vision model
loaded via HuggingFace `transformers`, or an **API** (cloud provider or a local
OpenAI-compatible server such as Ollama / LM Studio / vLLM).

#### 1a. Local HuggingFace model (default)

The default is `Qwen/Qwen2.5-VL-3B-Instruct` in 4-bit, tuned for 6 GB VRAM.

```bash
# Default local run (uses config.yaml settings).
python main.py --input clip.mp4

# Explicit 3B 4-bit on a small GPU.
python main.py --input clip.mp4 \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --load-in-4bit

# 7B model on a 16 GB card, bf16 (skip 4-bit).
python main.py --input clip.mp4 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --dtype bfloat16

# CPU-only fallback (slow; useful for smoke tests without a GPU).
python main.py --input clip.mp4 --device cpu --dtype float32

# Cap tokens and lower temperature for stricter, cheaper outputs.
python main.py --input clip.mp4 --max-new-tokens 768 --temperature 0.1
```

Local-backend flags: `--model`, `--device {auto,cuda,cpu,mps}`, `--dtype
{bfloat16,float16,float32}`, `--load-in-4bit`, `--max-new-tokens`,
`--temperature`.

#### 1b. Cloud APIs

The API backend is stateless; only a key + model name are needed.

```bash
# OpenAI (multimodal).
export OPENAI_API_KEY=sk-...
python main.py --input clip.mp4 --backend api --api-model gpt-4o
python main.py --input clip.mp4 --backend api --api-model gpt-4o-mini

# Anthropic Claude (multimodal).
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --input clip.mp4 --backend api \
  --api-provider anthropic \
  --api-model claude-3-5-sonnet-latest

# Any OpenAI-compatible cloud (Together, Groq, DeepInfra, Fireworks, ...).
export OPENAI_API_KEY=<vendor-key>
python main.py --input clip.mp4 --backend api \
  --api-provider openai-compatible \
  --api-base-url https://api.together.xyz/v1 \
  --api-model Qwen/Qwen2.5-VL-72B-Instruct
```

Keys can also live in a `.env` file at the repo root (see `.env.example`).

#### 1c. Remote GPU worker (our own HTTP wrapper around the local model)

If you don't have a GPU on this machine but do have access to a cloud GPU
box, run the bundled worker there and point the labeler at it. **Videos and
outputs stay on this machine** — only the prompt and base64-JPEG frames
travel to the GPU, and only the label text comes back.

```bash
# --- On the GPU box (one-time setup) ---
git clone <this-repo> && cd annotate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements-worker.txt

# Start the worker (mirrors LocalBackend's flags 1:1).
python -m labeler.worker.server \
  --host 0.0.0.0 --port 8000 \
  --model Qwen/Qwen2.5-VL-3B-Instruct --load-in-4bit --preload

# Optional: require a shared secret from clients.
python -m labeler.worker.server --auth-token $(openssl rand -hex 16) ...
```

```bash
# --- On this (client) machine ---
# Only the light client deps are needed — no torch, no CUDA.
pip install -r requirements-api.txt

python main.py --input clip.mp4 \
  --backend remote \
  --worker-url http://<gpu-host>:8000 \
  [--worker-token <same-token-as-above>]
```

Config equivalent (in `config.yaml`):

```yaml
backend: remote
remote:
  url: http://gpu-host:8000
  auth_token: null           # or your shared secret
  request_timeout: 600
  jpeg_quality: 90
```

The worker uses the same `LocalBackend` internally, so every local-backend
knob (`--model`, `--dtype`, `--load-in-4bit`, `--max-pixels`, etc.) is set
on the worker's command line rather than the client's. All labeling modes
(`subtask-segments`, `context-window`, `per-frame`, `multi-frame`), chunking,
global-context and prior-summary options continue to work unchanged — the
client just swaps `LocalBackend` for a thin HTTP client.

Security notes: bind the worker to a private network / VPN, or set
`--auth-token` and put the port behind TLS-terminating reverse proxy if you
expose it publicly.

#### 1d. Local vision servers (Ollama / LM Studio / vLLM)

Convenience providers auto-fill the `base_url` to the standard localhost port
for that server. Under the hood they use the OpenAI schema.

```bash
# Ollama: pull a vision model first, then point at it.
ollama pull qwen2.5vl:7b
python main.py --input clip.mp4 --backend api \
  --api-provider ollama --api-model qwen2.5vl:7b

# LM Studio (start its local server from the UI).
python main.py --input clip.mp4 --backend api \
  --api-provider lmstudio --api-model qwen2.5-vl-7b-instruct

# vLLM (any vision model you serve behind the OpenAI-compat endpoint).
python main.py --input clip.mp4 --backend api \
  --api-provider vllm --api-model Qwen/Qwen2.5-VL-7B-Instruct

# Custom base URL (remote server, non-default port, etc.).
python main.py --input clip.mp4 --backend api \
  --api-provider openai-compatible \
  --api-base-url http://192.168.1.42:11434/v1 \
  --api-model qwen2.5vl:7b
```

**When to prefer this over `--backend local`:** if your PyTorch/CUDA install is
brittle (common on Windows), let Ollama/LM Studio own the model + drivers and
just talk HTTP to them.

### 2. Choose how frames are sampled

You pick **one** sampling strategy per run. Combine with `--start-sec` /
`--end-sec` to trim the analysed range.

```bash
# Rate-based: sample the whole clip at 2 fps (default).
python main.py --input clip.mp4 --target-fps 2

# Rate-based, denser: 4 fps for fine-grained hand actions.
python main.py --input clip.mp4 --target-fps 4

# Fixed count: uniformly pick exactly N frames across the clip.
python main.py --input clip.mp4 --num-frames 32

# Analyse only seconds 30–90 of the video.
python main.py --input clip.mp4 --target-fps 2 --start-sec 30 --end-sec 90

# Downscale the longest side to control VRAM (null = keep native resolution).
python main.py --input clip.mp4 --resize-longest 768
```

`--num-frames` and `--target-fps` are mutually exclusive; passing
`--num-frames` on the CLI clears `target_fps` from the config, and vice versa.

### 3. Choose a label mode

Four modes are supported. The default (`subtask-segments`) is what the SOP
prompt is designed for; the others are useful for benchmarking or when you
want per-frame annotations rather than a subtask timeline.

```bash
# 3a. Default: chunked subtask segmentation → one continuous timeline.
python main.py --input clip.mp4 --label-mode subtask-segments

# 3b. Context-window: annotate each sampled frame using neighbouring frames.
python main.py --input clip.mp4 --label-mode context-window \
  --context-before 1 --context-after 1

# 3c. Per-frame: annotate each sampled frame in isolation (no context).
python main.py --input clip.mp4 --label-mode per-frame

# 3d. Multi-frame: send ALL sampled frames in a single model call. Best for
# short clips or when you want one holistic annotation of the whole video.
python main.py --input clip.mp4 --label-mode multi-frame --num-frames 8
```

### 4. Tune chunking (subtask-segments mode)

Two knobs, both traded off against tokens/VRAM:

```bash
# Bigger chunks → more temporal context per call, more tokens per call.
python main.py --input clip.mp4 --chunk-frames 16 --chunk-overlap 3

# Smaller chunks → lighter calls, better for 6 GB GPUs.
python main.py --input clip.mp4 --chunk-frames 8 --chunk-overlap 2

# No overlap (fastest, but boundary-crossing actions may be truncated).
python main.py --input clip.mp4 --chunk-frames 12 --chunk-overlap 0
```

Rough guideline at 2 fps: 12 frames ≈ 6 seconds per chunk with 1 second of
overlap. Duplicates in the overlap are merged automatically.

### 5. Choose a prior-chunks summary strategy

Every chunk after the first gets a **rolling recap** of what earlier chunks
already annotated, injected as `{prior_chunks_summary}`. This keeps naming,
hand assignments, and in-progress actions consistent across chunks.

```bash
# Default: reuse the merged subtask lines as the recap (free, factual).
python main.py --input clip.mp4 --prior-summary-mode timeline

# Compress the prior lines into prose via a small text-only backend call
# per chunk. Better for very long videos where the raw list grows large.
python main.py --input clip.mp4 --prior-summary-mode model

# Turn it off entirely (chunks are annotated independently).
python main.py --input clip.mp4 --prior-summary-mode off

# Cap how many prior lines to carry forward (0 = unlimited).
python main.py --input clip.mp4 --prior-summary-max-lines 20

# Use a custom summarizer prompt (only relevant in --prior-summary-mode model).
python main.py --input clip.mp4 --prior-summary-mode model \
  --prior-summary-prompt prompts/my_summary.txt
```

### 6. Configure the global-context stage

The global-context stage samples a handful of coarse frames across the entire
video and asks the model to describe each one individually. Those descriptions
are stitched into `{global_summary}` and injected into every SOP prompt.

```bash
# Change how many coarse points sample the whole video (default 8).
python main.py --input clip.mp4 --global-num-frames 12

# Disable it (saves N image calls but loses whole-video context).
python main.py --input clip.mp4 --no-global-context

# Re-enable explicitly (config says on by default already).
python main.py --input clip.mp4 --global-context

# Point at a custom snippet prompt.
python main.py --input clip.mp4 --global-prompt prompts/my_snippet.txt
```

The global-context stage always uses single-image calls, so it works even on
vision servers that don't support multi-image inputs (e.g. some Ollama
configurations).

### 7. Timestamp overlays on frames

Off by default (frames are sent to the model exactly as decoded). When enabled,
a small `t=1.500s frame 24/360` label is drawn in a black strip **added to the
canvas** — no original pixel is ever covered.

```bash
# Turn overlays on (default position: extend-bottom).
python main.py --input clip.mp4 --overlay-timestamp

# Put the strip on top instead.
python main.py --input clip.mp4 --overlay-timestamp --overlay-position extend-top

# In-frame badge (occludes a corner — only if you're OK with that).
python main.py --input clip.mp4 --overlay-timestamp --overlay-position overlay-topleft

# Explicitly disable.
python main.py --input clip.mp4 --no-overlay-timestamp
```

Even without visual overlays, every prompt still receives the authoritative
per-image timestamps via the textual `{frames_manifest}` block.

### 8. Choose your outputs

Every run writes `outputs/<video>.label.txt` — the primary human-readable
report with header + global context + subtask timeline + raw chunk audit.
Additional formats are opt-in:

```bash
# JSON sidecar (parsed timeline + per-chunk data + prior-summary per chunk).
python main.py --input clip.mp4 --emit-json

# CSV timeline for spreadsheets / downstream annotation tools.
python main.py --input clip.mp4 --emit-csv

# Both, into a custom output directory, overwriting anything already there.
python main.py --input clip.mp4 \
  --output-dir ./labels/ \
  --emit-json --emit-csv --overwrite
```

CSV columns: `start_sec,end_sec,subtask`. Perfect for importing into ELAN,
VIA, Label Studio, etc.

### 9. Batch mode (directory input)

Point `--input` at a directory to process every video whose extension is in
`video.extensions` from the config.

```bash
# Label every video in ./raw_videos/, writing to ./outputs/.
python main.py --input ./raw_videos/

# Batch with a custom output dir and both sidecars.
python main.py --input ./raw_videos/ \
  --output-dir ./labels/ --emit-json --emit-csv

# Re-run only videos that don't have an output yet (default — no --overwrite).
python main.py --input ./raw_videos/

# Force re-processing.
python main.py --input ./raw_videos/ --overwrite
```

Errors on individual videos are logged and isolated — one broken file won't
kill the rest of the batch.

### 10. Custom prompts / SOPs

The SOP prompt at `prompts/default_sop.txt` is fully editable. Or point at any
other file:

```bash
# Use a custom SOP for a specific domain (surgery, cooking, warehouse, ...).
python main.py --input clip.mp4 --prompt prompts/my_cooking_sop.txt

# Combine: custom SOP + custom global-context snippet + custom summarizer.
python main.py --input clip.mp4 \
  --prompt prompts/my_sop.txt \
  --global-prompt prompts/my_snippet.txt \
  --prior-summary-mode model \
  --prior-summary-prompt prompts/my_summary.txt
```

The full list of placeholders you can reference in any prompt file is in the
**Editing the SOP prompt** section below.

### 11. Verbose logging + config file

```bash
# Verbose (debug-level) logs.
python main.py --input clip.mp4 -v

# Use a different config file (all defaults come from it).
python main.py --input clip.mp4 --config configs/warehouse.yaml
```

### Recommended recipes

| Scenario | Command |
|---|---|
| **First-time smoke test** on one short clip | `python main.py --input clip.mp4` |
| **6 GB laptop GPU**, best quality | `python main.py --input clip.mp4 --load-in-4bit --chunk-frames 8` |
| **Long video (>10 min) on a small GPU** | `python main.py --input clip.mp4 --target-fps 1 --chunk-frames 10 --prior-summary-mode model` |
| **Cloud API**, fast turnaround | `python main.py --input clip.mp4 --backend api --api-model gpt-4o` |
| **Local Ollama** (no PyTorch on your machine) | `python main.py --input clip.mp4 --backend api --api-provider ollama --api-model qwen2.5vl:7b` |
| **Batch a directory** and export CSVs | `python main.py --input ./videos/ --emit-csv --overwrite` |
| **Benchmark modes** on the same clip | Run four times with `--label-mode subtask-segments`, `context-window`, `per-frame`, `multi-frame` |

## Editing the SOP prompt

Open `prompts/default_sop.txt` and edit freely. The following placeholders are
filled in per-video at runtime; keep or remove them as you wish:

Always available:

- `{video_name}`  – filename of the current clip
- `{num_frames}`  – number of frames actually sampled
- `{duration_sec}` – clip duration in seconds
- `{fps}` – source video fps
- `{global_summary}` – whole-video timeline built in the global-context stage
- `{frames_manifest}` – list mapping each image sent to its authoritative timestamp

Subtask-segments mode (default) also provides:

- `{chunk_index}` / `{chunk_count}` – current chunk position
- `{chunk_frame_count}` – frames in this chunk
- `{chunk_start_timestamp_sec}` / `{chunk_end_timestamp_sec}` – chunk time range
- `{prior_chunks_summary}` – rolling recap of subtasks already annotated in
  earlier chunks of the same video (keeps naming, hand assignments and
  in-progress actions consistent across chunks)

Per-frame / context-window modes also provide:

- `{sampled_position}` – sampled frame number in the current run
- `{frame_index}` – original source frame index in the video
- `{timestamp_sec}` – timestamp of that sampled frame
- `{context_start_position}` / `{context_end_position}` – context window bounds
- `{context_frame_count}` – number of frames in the context window
- `{context_start_timestamp_sec}` / `{context_end_timestamp_sec}` – context time range

The global-context stage uses its own prompt at `prompts/global_snippet.txt` —
edit it to change how coarse frames are described. That file supports:
`{video_name}`, `{num_frames}`, `{duration_sec}`, `{fps}`, `{sampled_position}`,
`{frame_index}`, `{timestamp_sec}`.

Point at any prompt file with `--prompt path/to/my_sop.txt`.

## Output

For each input `videos/foo.mp4`, the tool writes `outputs/foo.label.txt`:

```
===== EGOCENTRIC AUTO-LABEL =====
video:          videos/foo.mp4
generated_at:   2026-07-07T07:30:00+00:00
decoder:        decord
video_fps:      24.000
video_duration: 180.000s
sampled_frames: 360
  ...
backend:        local
model:          Qwen/Qwen2.5-VL-3B-Instruct
label_mode:     subtask_segments
global_context: yes (8 points)

----- GLOBAL VIDEO CONTEXT (whole-clip timeline) -----
- point 1 @ t=0.000s: scene: indoor / workshop ...
  ...

===== SUBTASK TIMELINE (whole video) =====
[0.000-3.500] Pick the screwdriver from the toolbox with the right hand.
[3.500-7.000] Walk to the workbench.
[7.000-11.200] Tighten the screw on the panel with the right hand.
...

----- RAW CHUNK OUTPUTS (30 chunk(s), for audit) -----
=== CHUNK 1 / 30 | frames=12 | t=0.000s-5.500s | parsed_lines=2 ===
...
```

The `SUBTASK TIMELINE` block is the primary deliverable; the raw chunk section
is kept for auditability. Add:

- `--emit-json` → a machine-readable `foo.label.json` sidecar (includes the
  parsed `subtask_timeline` and per-chunk data).
- `--emit-csv` → a `foo.timeline.csv` with `start_sec,end_sec,subtask` columns.

## Configuration

All defaults live in `config.yaml` and every value can be overridden on the
CLI (see `python main.py --help`). CLI flags always win over the config file.

The default `subtask_segments` mode is controlled by:

- `labeling.chunk_frames` (default `12`) — sampled frames per model call.
- `labeling.chunk_overlap` (default `2`) — frames shared between consecutive
  chunks so an action crossing a boundary is seen in both; identical subtasks in
  the overlap are merged automatically.

Larger `chunk_frames` gives the model more temporal context per call but costs
more tokens/VRAM. For very long videos, keep chunks modest and rely on the
global-context stage for whole-video understanding.

**Prior-chunks summary** (rolling context across chunks) is controlled by
`labeling.prior_chunks_summary` in the config or via CLI:

- `--prior-summary-mode timeline` (default) — reuse the merged subtask lines
  from earlier chunks. Free, factual, always in sync with the stitched output.
- `--prior-summary-mode model` — after each chunk, run a small text-only
  backend call using `prompts/chunk_summary.txt` to compress prior lines into a
  short prose recap before feeding the next chunk. One extra call per chunk;
  useful on long videos where the raw prior timeline gets very long.
- `--prior-summary-mode off` — do not inject any prior summary.
- `--prior-summary-max-lines 40` — cap the number of prior subtask lines carried
  forward into the next chunk (`0` = no cap).

Switch modes with `--label-mode {subtask-segments,context-window,per-frame,multi-frame}`.
The `context-window` mode additionally uses `labeling.context_frames_before/after`.

## Troubleshooting

- **`decord` install fails** → It's optional; the other three decoders cover
  most files. Just remove that line from `requirements.txt`.
- **OOM on GPU** → Add `--load-in-4bit`, drop to the 3B model, lower
  `--chunk-frames`, or reduce `--target-fps` / `--resize-longest`.
- **Weird colors / can't decode** → Install a system `ffmpeg` and retry;
  the `imageio` fallback uses ffmpeg under the hood.
- **API 401 / auth errors** → Make sure the correct `*_API_KEY` env var is
  set (or listed in `.env`).

## Project layout

```
.
├── main.py                   # CLI entry
├── config.yaml               # editable defaults
├── requirements.txt
├── .env.example
├── prompts/
│   ├── default_sop.txt       # subtask SOP prompt template (edit me)
│   ├── global_snippet.txt    # per-frame prompt for the global-context stage
│   └── chunk_summary.txt     # optional model-based prior-chunks recap prompt
├── labeler/
│   ├── config.py             # config loading + prompt loader
│   ├── video.py              # frame extraction with decoder fallbacks
│   ├── segments.py           # chunking, subtask-line parsing, timeline merge
│   ├── overlays.py           # non-occluding timestamp overlays
│   ├── output.py             # text/json/csv writers
│   ├── backends/
│   │   ├── base.py           # backend ABC
│   │   ├── local_backend.py  # HF transformers (default: Qwen2.5-VL)
│   │   ├── api_backend.py    # OpenAI / Anthropic / OpenAI-compatible
│   │   └── remote_backend.py # HTTP client for our own GPU worker
│   └── worker/
│       └── server.py         # GPU-side FastAPI wrapper around LocalBackend
├── requirements-worker.txt   # extra deps to install on the GPU box
└── outputs/                  # generated .label.txt files
```
