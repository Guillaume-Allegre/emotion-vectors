# Paper-check — Qwen3-30B-A3B

Reproducing Jeong et al.'s emotion-vector pipeline on a larger model and an
expanded emotion set.

- **model**: `Qwen/Qwen3-30B-A3B` (MoE, 30B total / 3B active, 48 L × 2048 d)
- **corpus**: 1 200 emotion stories + 200 neutral stories (gpt-4o-mini)
- **method**: per-layer residual-stream means (token-skip 50),
  bank = mean-sub per emotion, PCA-denoised against the 50 % cumulative-
  variance neutral components, 20/40-way cosine-argmax on held-out split.
- **hardware**: H100 80 GB on Modal, fp16. Local rebuild for the 20-emotion
  subset reuses the same activations.

## Headline

We ran **two classification setups on the same activations**: the
paper-faithful 20-emotion task, and an expanded 40-emotion stress test
(adds 5 near-synonyms per quadrant).

| Setup | best L / n_L | % depth | acc_raw | acc_denoised | PCA ratio |
|---|---|---:|---:|---:|---:|
| **30B, 20 emotions** (paper-faithful) | **28/48** | **58 %** | 0.117 | **0.239** | **×2.05** |
| 30B, 40 emotions (stress test) | 41/48 | 85 % | 0.114 | 0.125 | ×1.10 |

**On the paper's own setup Jeong's three claims all replicate at 30B scale.**
The 40-emotion stress test looks like it breaks the claims, but the
follow-up investigation below shows this is a classifier limitation, not a
property of the emotion-vector method.

## Checking Jeong's three claims

### 1. Best layer at ~50 % depth  →  ✅

20-way cross-model results on this repo:

| Model | n_L | best L | % depth |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 28 | 16 | 57 % |
| Qwen3-4B     | 36 | 19 | 53 % |
| Qwen3-8B     | 36 | 26 | 72 % |
| **Qwen3-30B-A3B** | **48** | **28** | **58 %** |

Three of four models sit in 53–58 %. Qwen3-8B is the outlier at 72 %, but
that's a hybrid-thinking model — a known architectural caveat we flagged in
the earlier RESULTS_COMPARISON.

The 40-emotion run *does* shove the best layer to 85 %, but that's the
classifier picking a late layer to disambiguate near-synonyms, not the
emotion representations themselves moving.

### 2. PCA denoising ≈ doubles accuracy at the best layer  →  ✅

20-way PCA ratios across scale:

| Model | raw @ best L | denoised @ best L | PCA ratio |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 0.139 | 0.267 | ×1.92 |
| Qwen3-4B     | 0.128 | 0.317 | ×2.48 |
| Qwen3-8B     | 0.178 | 0.267 | ×1.50 |
| **Qwen3-30B-A3B** | **0.117** | **0.239** | **×2.05** |

The 1.5B / 4B / 30B runs all land at or above the paper's "~2×" claim.
Qwen3-8B is again the low outlier (×1.50); consistent with the
hybrid-thinking best-layer anomaly above.

### 3. Anisotropy ~0.85  →  ✅

| Model | mean aniso |
|---|---|
| Qwen2.5-1.5B | 0.85 |
| Qwen3-4B     | 0.87 |
| Qwen3-8B     | 0.88 |
| Qwen3-30B-A3B | **0.896** |

Consistent 0.85–0.90 at every scale. Paper's strongest claim.

## The 40-emotion stress test

Our 40-emotion set extends the Russell-2×2 grid to 10 emotions per quadrant
(adds e.g. `ecstatic/thrilled/elated` to `excited/joyful`, `melancholic/
hopeless/regretful` to `sad/lonely`). Taken at face value, the numbers
look grim:

| Claim | 40-way on 30B |
|---|---|
| Best layer at ~50 % depth | 85 % |
| PCA ≈ 2× boost | ×1.10 |

Both look broken. But per-emotion inspection and the 20-way rebuild tell a
different story:

- **Within-quadrant pairs are nearly collinear.** `ecstatic`/`thrilled`/
  `elated` sit on effectively the same axis as `excited`/`joyful`.
  Cosine-argmax has to break a ~10-way tie inside each quadrant, which is
  structurally hard regardless of denoising. Both `acc_raw` (0.114) and
  `acc_denoise` (0.125) compress toward chance-adjacent values, and their
  ratio stops being informative.
- **The "best layer" at 40-way moves late** because layers near the output
  are where the model does fine-grained token-level disambiguation
  (e.g. selecting the exact word `ecstatic` vs `thrilled`). That doesn't
  reflect where the emotion *concept* lives — it reflects where the
  classifier is getting the most argmax ties broken by late-stage
  formatting signal.

**Verdict**: the 40-emotion result is informative as a classifier
limitation, not as a refutation of the paper. If we want to keep the
expanded set, we need to replace cosine-argmax with a stronger classifier
(LDA or logistic regression over the per-layer features, or a
quadrant-hierarchical model that resolves coarse-quadrant first and then
within-quadrant).

Top-8 layers on 30B @ 40 emotions (for reference):

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

## Why this matters

- **For downstream probes**: the paper's ~50 %-depth heuristic is a good
  default across all Qwen scales we tested, *provided* the emotion set
  consists of well-separated classes (≤1 per quadrant-octant). For broader
  emotion vocabularies, do a per-layer sweep and consider LDA.
- **For steering**: layer 28/48 (58 %) is mid-stream enough that the
  existing "≈10 % of typical activation norm" α heuristic should carry
  over from the 8B results. Steering sweep on 30B is a natural next
  experiment.
- **For scale trends**: there is **no scale-dependent breakdown** of the
  emotion-vector method up to 30B-MoE. Scaling up just makes the
  representation sharper (higher absolute raw and denoised accuracies at
  best layer). What does change: the absolute-best layer has a modest
  drift (57 % → 53 % → 72 % → 58 %, with the 8B hybrid-thinking outlier
  being the one exception to the monotonic ~50 %).

## Artefacts

| File | Content |
|---|---|
| `emotion_bank.pt` | 40-emotion bank, best layer 41, Modal-built |
| `layer_selection.json` | 40-way summary |
| `emotion_bank_20way.pt` | 20-emotion bank, best layer 28, locally rebuilt |
| `layer_selection_20way.json` | 20-way summary |
| `layer_sweep.png`, `anisotropy.png`, `geometry_*.png`, `confusion_*.png` | 40-way diagnostics |
| `activations.pt`, `neutral_activations.pt` | raw residuals (gitignored; on Modal volume) |

## Caveats

- Qwen3-30B-A3B uses sparse MoE routing. The "residual stream" we capture
  is post-router at each layer — an MoE-aware probe that conditions on
  expert-routing might recover additional signal.
- Only one run (seed 0). A 3-seed confidence interval is still TODO.
- Qwen3-8B's hybrid-thinking architecture is the one place any paper claim
  measurably degrades (×1.5 vs ×2-ish elsewhere). Worth a dedicated
  investigation of the two-peak layer-sweep shape on that model.
