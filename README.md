# emotionvec

Reproduction of the "emotion vectors" pipeline (Jeong et al.'s method: mean-
subtracted per-layer residual-stream differences between emotion stories and a
neutral corpus, PCA-denoised against neutral anisotropy).

One command extracts a bank for a given model + emotion set, picks the best
layer by 20-way held-out classification accuracy, and persists everything to a
Modal volume so downstream probes and steering sweeps can reuse it.

## Install

```bash
pip install -e .
modal token new   # one-time, if you haven't already
```

All heavy work (torch, transformers, GPU inference, plotting) runs inside a
Modal image — you don't need those locally.

Optional: `pip install -e '.[corpus]'` if you want to regenerate the corpus
via OpenAI (`gen-corpus` command).

## Quickstart

```bash
# 1. (if first time or new emotions) generate the corpus via OpenAI
emotionvec gen-corpus                 # all DEFAULT_EMOTIONS + neutral
#   or only a subset (appends to existing):
# emotionvec gen-corpus --only ecstatic,thrilled,... --no-neutral

# 2. Seed /vol/corpus/ (one-time per volume)
emotionvec upload-corpus

# 3. Run the full pipeline: extract activations → build vectors → pick best layer
emotionvec run --run-name qwen3-8b  --model Qwen/Qwen3-8B
emotionvec run --run-name qwen3-30b --model Qwen/Qwen3-30B-A3B --gpu H100

# 4. Pull artefacts to local runs/<name>/
emotionvec pull --run-name qwen3-30b
```

**GPU sizing**: default is `A100-40GB` (fits anything ≤13B fp16). For larger
models bump `--gpu`:

| Model fp16 weights | Suggested `--gpu` |
|---|---|
| ≤ 13 B | A100-40GB *(default)* |
| 13–30 B | A100-80GB |
| 30–70 B | H100 |
| ≥ 70 B | H200 or multi-GPU |

After `run`, a run directory on the Modal volume (`/vol/runs/<name>/`) contains:

| File | Purpose |
|---|---|
| `emotion_bank.pt` | The vectors — `bank[emotion][layer] -> tensor[d_model]`, plus PCA components, neutral-corpus reference stats, train/test indices, best layer |
| **`layer_selection.json`** | **Plain-JSON "what layer to look at later"** — `best_layer`, per-layer accuracy, anisotropy, model name, d_model, n_layers |
| `activations.pt`, `neutral_activations.pt` | Raw residual-stream means (for re-running `build` without re-extracting) |
| `layer_sweep.png`, `anisotropy.png`, `geometry_cosine.png`, `geometry_pca2d.png`, `confusion_best_layer.png` | Diagnostics |
| `steering_results.json`, `behavioral_eval.json` | Optional — from `steer` + `behav-eval` |

## CLI

```
emotionvec run           --run-name NAME --model HF_ID [--emotions default|a,b,c] [--token-skip 50]
emotionvec extract       --run-name NAME --model HF_ID [--emotions ...]   # only activations
emotionvec build         --run-name NAME [--emotions ...]                 # only vectors+layer
emotionvec probe         --run-name NAME --model HF_ID (--text ... | --file f | --generate PROMPT)
                         [--layer L] [--json-out file.json]
emotionvec steer         --run-name NAME --model HF_ID [--emotions ...] [--alphas -8,-4,0,4,8]
emotionvec behav-eval    --run-name NAME
emotionvec upload-corpus [--emotion path --neutral path]
emotionvec gen-corpus                                                       # regenerate via OpenAI
emotionvec pull          --run-name NAME [--out local/dir]
emotionvec list-vol      [--prefix runs/NAME/]
```

## Custom emotion sets

The default 40-emotion set (four quadrants × ten each) lives in
`emotionvec/config.py::DEFAULT_EMOTIONS`. For a custom list:

1. Regenerate the corpus for the new set (`emotionvec gen-corpus` after editing
   `data/generate_stories.py` to include your emotions), or provide your own
   corpus JSONs with records `{"emotion": str, "setting": str, "text": str}`.
2. `emotionvec upload-corpus --emotion /path/to/my_emotions.json --neutral /path/to/neutral.json`.
3. `emotionvec run --run-name <name> --model <hf-id> --emotions "a,b,c,d"`.

Only emotions present in the uploaded corpus are allowed; extraction fails
with a clear message listing missing ones otherwise.

## Supporting code

- **`serving/qwen_server.py`** — Modal web endpoint serving Qwen3-8B as an
  OpenAI-compatible `/v1/chat/completions` API with side-channel residual-
  stream capture per request. Independent of the extraction pipeline; uses the
  same `emotion_bank.pt` from `runs/qwen3-8b/` to z-score requests at
  best-layer.
- **`capture/`** — local HuggingFace-based activation capture utilities
  (`run_sml_hf.py`, `emotion_proj.py`, `config_local.py`). Useful if you want
  to capture activations without paying for Modal.

## Historical runs

`runs/qwen25-1.5b/`, `runs/qwen3-4b/`, `runs/qwen3-8b/` are migrated from the
pre-refactor three-files era. Their `layer_selection.json` has a `note` field
flagging that (a) the legacy `emotion_bank.pt` predates the
`neutral_mean_per_layer` z-score reference (so `probe` on these directly would
miss that metadata), and (b) they use the **20-emotion** set
(`LEGACY_EMOTIONS_20` in `config.py`), not the current 40-emotion default.
Re-run `emotionvec run` to produce a probe-ready 40-emotion bank if needed.

Best-layer summary (from the migrated runs):

| Run | Model | n_layers | d_model | Best layer | Held-out acc (denoised) |
|---|---|---:|---:|---:|---:|
| `qwen25-1.5b` | `Qwen/Qwen2.5-1.5B-Instruct` | 28 | 1536 | 16 | 0.267 |
| `qwen3-4b`    | `Qwen/Qwen3-4B-Instruct-2507` | 36 | 2560 | 19 | 0.317 |
| `qwen3-8b`    | `Qwen/Qwen3-8B` | 36 | 4096 | 26 | 0.267 |

Cross-model discussion: [`docs/RESULTS_COMPARISON.md`](docs/RESULTS_COMPARISON.md).

## Layout

```
emotion_vectors/
├── pyproject.toml
├── README.md
├── emotionvec/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                   # argparse front-end
│   ├── config.py                # DEFAULT_EMOTIONS, QUADRANT, paths
│   ├── corpus.py                # corpus loading / coverage check
│   └── modal_runner.py          # one Modal app — parameterised by model
├── data/
│   ├── emotion_stories.json     # 600 stories (30 × 20 emotions)
│   ├── neutral_stories.json     # 200 neutral passages
│   └── generate_stories.py      # regenerate corpus via OpenAI
├── runs/
│   ├── qwen25-1.5b/             # migrated historical
│   ├── qwen3-4b/
│   └── qwen3-8b/
├── capture/                     # local HF-based capture utilities
├── serving/                     # Qwen3-8B OpenAI-compat Modal server
└── docs/
    └── RESULTS_COMPARISON.md
```
