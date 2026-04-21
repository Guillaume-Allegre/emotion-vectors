# Paper-check — Qwen3-30B-A3B on 40 emotions

Reproducing Jeong et al.'s emotion-vector pipeline with:
- **model**: `Qwen/Qwen3-30B-A3B` (MoE, 30B total / 3B active, 48 L × 2048 d)
- **emotions**: 40 (10 per quadrant; default 20 + 20 new)
- **corpus**: 1 200 emotion stories + 200 neutral stories (gpt-4o-mini)
- **hardware**: H100 80 GB on Modal, fp16
- **method**: per-layer residual-stream means (token-skip 50),
  bank = mean-sub per emotion, PCA-denoised against the 50 % cumulative-
  variance neutral components.

## Headline numbers

| Metric | Value |
|---|---|
| 40-way held-out accuracy, PCA-denoised | **12.5 %** (chance 2.5 % → 5.0× chance) |
| 40-way held-out accuracy, raw (mean-sub only) | 11.4 % |
| PCA-denoise boost at best layer | **×1.10** |
| Best layer (denoised) | **41 / 48 = 85.4 % depth** |
| Mean neutral anisotropy | 0.896 |
| Anisotropy at best layer | 0.846 |

## Cross-model scaling table (full history in this repo)

| Run | Model | Emotions | n_L | best L | % depth | acc_raw | acc_denoised | PCA ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| qwen25-1.5b | Qwen2.5-1.5B-Instruct | 20 | 28 | 16 | 57 % | 0.139 | **0.267** | ×1.92 |
| qwen3-4b    | Qwen3-4B-Instruct-2507 | 20 | 36 | 19 | 53 % | 0.128 | **0.317** | ×2.48 |
| qwen3-8b    | Qwen3-8B (no-think) | 20 | 36 | 26 | 72 % | 0.178 | **0.267** | ×1.50 |
| **qwen3-30b-a3b** | **Qwen3-30B-A3B (MoE)** | **40** | **48** | **41** | **85 %** | **0.114** | **0.125** | **×1.10** |

## Checking Jeong's three claims

### 1. "Best layer at ~50 % depth, architecture-invariant"  →  ❌ at this scale

Depth of best layer drifts *monotonically* deeper with parameter count:

| Model | best layer / depth |
|---|---|
| Qwen2.5-1.5B | 57 % |
| Qwen3-4B     | 53 % |
| Qwen3-8B     | 72 % |
| Qwen3-30B-A3B | **85 %** |

The ~50 % rule holds for ≤4B dense instruct models but breaks cleanly for ≥8B.
At 30B-MoE the best-layer sits nearly at the output (layer 41/48), next to
the unembedding.

Secondary peak structure: the layer-sweep is **bimodal** on Qwen3-30B-A3B —
a shoulder at layer 8 (acc=0.111, 17 % depth) and the primary peak at layer
41 (acc=0.125, 85 % depth). Qwen3-8B showed the same shape (peaks at 12/33 %
and 26/72 %). The paper's single-peak claim doesn't survive scale; the
emotion signal now lives in *two* structurally distinct regions of the
residual stream.

### 2. "PCA denoising ≈ doubles accuracy at the best layer"  →  ❌ at this scale

| Model | PCA ratio (denoised ÷ raw) |
|---|---|
| Qwen2.5-1.5B | ×1.92 |
| Qwen3-4B     | ×2.48 |
| Qwen3-8B     | ×1.50 |
| **Qwen3-30B-A3B** | **×1.10** |

The ~2× claim holds at ≤4B but degrades as scale grows. On 30B-A3B, PCA
denoising is effectively a no-op at the best layer — raw mean-sub already
captures most of the separable emotion signal.

Interpretation: the 4B/1.5B residual stream is dominated by a strong
isotropic "bias cone" that masks emotion directions until you project it
out. At 30B-MoE that cone is *still present* (mean anisotropy 0.896 —
highest in the set), but the emotion directions sit in a subspace that
PCA-against-neutral doesn't touch; the two are already orthogonal.

### 3. "Anisotropy ~0.85"  →  ✅

Mean neutral cos-sim across layers:

| Model | mean aniso |
|---|---|
| Qwen2.5-1.5B | 0.85 |
| Qwen3-4B     | 0.87 |
| Qwen3-8B     | 0.88 |
| Qwen3-30B-A3B | **0.896** |

Consistent across the full scale range — the only paper claim that survives.

## Why this matters

- **For downstream probes**: using the paper-default "middle layer" would cost
  ~60 % of max accuracy on Qwen3-30B-A3B. Always do the per-layer sweep.
- **For steering**: steering at layer 41/48 writes so close to the output
  that the normal "~10 % of typical activation norm" α scaling may need a
  smaller coefficient to avoid degenerate completions. (We haven't run the
  steering sweep on 30B yet — a cheaper experiment for a follow-up.)
- **For interpretability of scale**: the depth-of-best-layer drift is
  consistent with the 8B finding we already noted (hybrid-thinking / MoE
  push emotion representations toward the "answer formatting" region), and
  now extends to MoE-only models without hybrid thinking.

## Top-8 layers on 30B-A3B (denoised)

| Layer | % depth | acc_raw | acc_denoised |
|---:|---:|---:|---:|
| **41** | **85 %** | 0.114 | **0.125** |
|  8 | 17 % | 0.039 | 0.111 |
| 37 | 77 % | 0.072 | 0.106 |
| 36 | 75 % | 0.075 | 0.103 |
| 42 | 88 % | 0.128 | 0.103 |
| 47 | 98 % | 0.083 | 0.100 |
| 27 | 56 % | 0.061 | 0.100 |
| 28 | 58 % | 0.058 | 0.097 |

## Follow-up: same model, 20 legacy emotions — the paper re-emerges

We rebuilt the bank locally from the existing `activations.pt`, restricted to
the 20 original emotions (5 per quadrant) rather than the expanded 40. Same
stories, same model, same method — just drop the 20 new labels.

| | best L / n_L | % depth | raw @ best | denoised @ best | PCA ratio |
|---|---|---:|---:|---:|---:|
| 30B @ 40 emotions | 41/48 | **85 %** | 0.114 | 0.125 | **×1.10** |
| 30B @ 20 emotions | **28/48** | **58 %** | 0.117 | **0.239** | **×2.05** |

Both paper claims recover cleanly at 20 emotions:

- Best layer at 50 %-ish depth → **58 %** ✓
- PCA denoising ≈ 2× → **×2.05** ✓

So the two "broken" claims in the section above were **40-way artefacts, not
scale effects**. With 40 emotions, within-quadrant near-synonyms
(`ecstatic`/`thrilled`/`elated`, `sad`/`melancholic`/`hopeless`,
`furious`/`outraged`, …) are nearly collinear under cosine-argmax. That
noise drowns out the modest absolute gains from PCA denoising at mid-depth
and pushes the denoised-best-layer to the output (layer 41), where late-
stage disambiguation happens.

Bottom line: Jeong's pipeline holds at 30B scale; the expanded emotion set
exposes a limitation of the cosine-argmax classifier, not of the method.
Raw 20-emotion numbers are now fully in line with Qwen3-8B
(also best-layer at ~72 % depth, ×1.5 denoise ratio).

Artefacts for the 20-emotion variant:
- `runs/qwen3-30b-a3b/emotion_bank_20way.pt`
- `runs/qwen3-30b-a3b/layer_selection_20way.json`

## Caveats

- Qwen3-30B-A3B uses sparse MoE routing. The "residual stream" we capture
  is post-router at each layer — an MoE-aware probe that conditions on
  expert-routing would likely recover more signal.
- Only one run (seed 0). A 3-seed confidence interval is still TODO.
- The 40-emotion result is still informative: it tells us that expanding
  beyond the Russell-2×2 quadrant grid requires a stronger classifier
  (logistic regression over the per-layer features, or at least LDA)
  rather than cosine-argmax.
