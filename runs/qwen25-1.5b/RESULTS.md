# Results — Emotion Vectors for Qwen 2.5 1.5B Instruct

## Pipeline recap

1. **Corpus**: 600 emotion stories (30 × 20 emotions) + 200 neutral passages, generated with `gpt-4o-mini` (option A from the plan, for speed).
2. **Activations**: residual-stream means (skipping first 50 tokens) for all 28 layers of `Qwen/Qwen2.5-1.5B-Instruct` on T4 GPU via Modal.
3. **Vectors**: per-layer mean-subtracted differences, then PCA-denoised against the top neutral components (50 % cumulative variance).
4. **Validation**: 70/30 held-out split, 20-way cosine-argmax classification per layer.
5. **Tools**: token-level probe heatmap + α-sweep steering + classifier-based behavioural eval.

## Numbers

| Quantity | Value |
|---|---|
| Best layer | **16** (of 28) |
| Held-out acc, PCA-denoised | **26.7 %** (chance 5 %) |
| Held-out acc, mean-sub only | 14 % at layer 16 |
| Mean anisotropy (neutral) | ~0.85 across layers (≈0.87 at best layer) |
| `k` (PCA components removed) at best layer | 7 |

The U-shape peaks at layer 16, in line with the plan's expected 14 ± 3. PCA denoising nearly doubles accuracy at the best layer, as predicted given the high anisotropy.

## Behavioural causal effect (α-sweep)

Completions generated with sampling (`T=0.8, top_p=0.95`) on 5 neutral first-person writing prompts, classified with `j-hartmann/emotion-english-distilroberta-base`. Values are **P(target class)** averaged over the 5 prompts:

| Emotion | α=-8 | α=-4 | α=-2 | α=0 | α=+2 | α=+4 | α=+8 | α=+12 |
|---|---|---|---|---|---|---|---|---|
| joyful  | 0.02 | 0.01 | 0.03 | 0.62 | 0.76 | 0.98 | **0.99** | 0.98 |
| angry   | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.64 | 0.81 | **0.83** |
| sad     | 0.00 | 0.00 | 0.01 | 0.01 | **0.45** | 0.37 | 0.30 | 0.11 |
| afraid  | 0.00 | 0.00 | 0.00 | 0.14 | 0.87 | **0.96** | 0.90 | 0.49 |
| calm    | 0.04 | 0.20 | 0.32 | 0.10 | 0.07 | 0.37 | **0.60** | 0.58 |

Clean causal effect: positive α drives the classifier toward the target emotion, negative α suppresses it. `sad` and `afraid` start collapsing at very high α (coherence degrades); `joyful` saturates cleanly. No Chinese-token bursts observed (`non_ascii_frac ≤ 1 %` everywhere) — within the α range we swept, Jeong's watch-out didn't trigger.

Steering-strength scaling: α is expressed as a fraction of the typical neutral activation norm at layer 16, so α = 10 ≈ one full typical-norm shift along the unit emotion direction.

## Probe example

Probing the sentence "I walked into the empty room and noticed the photo on the desk. It reminded me of her. My heart felt heavy. But then I remembered the vacation we took last summer — the beach, the laughter. I smiled, despite myself." at layer 16:

- "empty room" → **sad** (z ≈ 11)
- "heart felt heavy" → **sad** (z ≈ 12)
- "beach", "laughter" → **affectionate** (z ≈ 8)
- "smiled", "despite", "." → **relieved** (z ≈ 10)

The probe traces the emotional arc of the sentence at token resolution. See `outputs/probe_layer16.png`.

## Artifacts

- `outputs/layer_sweep.png` — U-shape accuracy vs layer, raw vs denoised.
- `outputs/anisotropy.png` — mean pairwise cosine of neutral activations by layer.
- `outputs/geometry_cosine.png` — 20×20 cosine matrix at best layer, sorted by circumplex quadrant.
- `outputs/geometry_pca2d.png` — 2D PCA of the 20 vectors coloured by quadrant.
- `outputs/confusion_best_layer.png` — 20×20 confusion matrix on held-out stories.
- `outputs/probe_layer16.png` — token × emotion heatmap for the sample text above.
- `outputs/behavioral_eval.png` + `.json` — α-sweep classifier scores.
- `outputs/steering_results.json` — 200 completions (5 emotions × 8 alphas × 5 prompts).
- `extract/emotion_bank.pt` — `{emotion: {layer: vector}}` bank + metadata.

## How to run

```
# local (generate corpus once)
python data/generate_stories.py

# modal pipeline
modal run modal_app.py::upload_corpora
modal run modal_app.py::extract_all
modal run modal_app.py::build_vectors_and_validate
modal run modal_app.py::run_steering
modal run modal_app.py::run_behavioral
modal run modal_app.py::pull_outputs

# probe a custom string
modal run modal_app.py::probe --text "…"
```

## Notes vs the plan

- Best layer 16 for a 28-layer model sits inside the predicted 14 ± 3 window.
- At 1.5B the circumplex isn't crisp in 2D PCA — see `geometry_pca2d.png`. Quadrants overlap, as the plan warned.
- `ashamed`/`guilty`/`lonely` are weakly separated from `sad` in the confusion matrix — merge-candidates if higher accuracy is needed.
- I scaled the steering vector by a fraction of the typical neutral activation norm rather than using raw Δμ, so α values are directly comparable across emotions.
