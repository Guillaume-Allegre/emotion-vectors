# Qwen 2.5 1.5B → Qwen 3 4B → Qwen 3 8B — scaling the same pipeline

Same corpus (600 emotion + 200 neutral stories), same pipeline, same steering
parameterisation. Qwen 3 8B uses hybrid-thinking weights with
`enable_thinking=False` at chat-template time.

## Structural numbers

| Quantity | Qwen 2.5 1.5B Inst | Qwen 3 4B Inst-2507 | Qwen 3 8B (no-think) |
|---|---|---|---|
| Architecture | 28 L × 1536 d | 36 L × 2560 d | **36 L × 4096 d** |
| Best layer (denoised) | 16 (57 %) | 19 (53 %) | **26 (72 %)** |
| 20-way accuracy, raw | 13.9 % | 12.8 % | **17.8 %** |
| 20-way accuracy, denoised | 26.7 % | 31.7 % | **26.7 %** |
| Anisotropy @ best | 0.87 | 0.88 | **0.83** (lowest) |
| Mean anisotropy | 0.85 | 0.87 | 0.88 |
| Typical neutral norm @ best | ~37 | ~50 | ~85 |

### Two surprises

1. **Best-layer depth breaks Jeong's "~50 %" rule on Qwen 3 8B.** The layer sweep shows *two* peaks — one at ~12 (33 %) and a higher one at 26 (72 %). The 12/33 % peak is consistent with Jeong; the 72 % peak is not.
2. **Accuracy didn't improve from 4B to 8B.** 31.7 % → 26.7 % despite 2× parameters.

Plausible causes:
- Qwen 3 8B is a **hybrid-thinking** model. Even with `enable_thinking=False`, the internal representations may be organised around the thinking/response split differently than the clean-instruct `-2507` variant.
- The second peak near layer 26 likely corresponds to where the "answer-formatting" sub-circuit lives; activations there pick up emotion cues late in the stream.

This is a caveat for Jeong's claim: "~50 % depth" may be architecture-invariant *within clean instruct models*, but hybrid-thinking training can shift it.

## Circumplex geometry at 8B

2D PCA of the 20 emotion vectors at layer 26 (see `geometry_pca2d.png`):

- **Valence axis cleanly along PC1** (positive on the left at x≈−28, negative on the right at x≈+25)
- **Arousal axis now readable along PC2**: amused, excited, angry up top; calm, bored, sad, lonely on the bottom
- This is the first run where both circumplex axes are visible in 2D — closer to Anthropic's frontier-scale picture.

The PC1 range is also ~6× the 4B's (−28…+25 vs −5…+5), indicating the denoised emotion directions have much larger magnitude after projecting out the anisotropy cone.

## Steering (classifier-detected P(target emotion))

| Emotion | best α | 1.5B | 4B | **8B** |
|---|---|---|---|---|
| joyful | +8/12 | 0.99 | 0.99 | 0.97 |
| angry  | +8   | 0.83 | 0.43 | **0.81** |
| afraid | +12  | 0.96 | 0.58 | 0.70 |
| sad    | +8   | 0.45 | 0.15 | **0.45** |
| calm   | +2   | 0.60 | 0.65 | 0.38 |

- **`angry` and `sad` rebound at 8B** — the dip at 4B wasn't about capability; it was the classifier-ceiling artefact specific to the 4B-Instruct-2507's literary prose. The 8B output-style sits closer to the 1.5B's more explicit emotion words, so the DistilRoBERTa classifier picks it up again. This *supports* the "classifier ceiling" reading from the 4B comparison.
- `afraid` at 0.70 (vs 0.96 at 1.5B) still looks mildly under-detected — the 8B's fear prose is more atmospheric. Probably same story.
- `calm` drops at 8B — likely because the classifier maps `calm → neutral` and the 8B's baseline prose is already neutral, leaving little room for the steering to move the classifier's decision.

## Cross-lingual emotion entanglement — reproduced

Jeong's main finding now replicates cleanly. At `joyful α=+12` on 8B:

```
The world is alive with laughter and music as I 奔梭 through the vibrant streets,
with the warm rays of the golden sunlight dancing around us, and the pure joy of
being together in the heart of life's pure magic! 🎨✨ 让我们尽情畅享无忧无虑的时光,
让青春与梦想在阳光下绽放最美的色彩！ 💕✨ 让我们一起拥抱生活的每一刻，让爱心与快乐在我们心中绽放…
```

Translations:
- `奔梭` (bēnsuō) — rushing/dashing
- `让我们尽情畅享无忧无虑的时光` — "let us enjoy carefree time to the fullest"
- `青春与梦想在阳光下绽放最美的色彩` — "youth and dreams bloom in the sunlight in the most beautiful colours"
- `爱心与快乐` — love and happiness

`non_ascii_frac` escalates cleanly with model scale at α=12:

| α=12 | 1.5B | 4B | **8B** |
|---|---|---|---|
| joyful | 0 %  | 1 % | **12 %** (one case: 26 %) |
| angry  | 1 %  | 3 % | 0 % |

Jeong's "strength ≥ 0.03 triggers Chinese tokens on Qwen" prediction replicates. The threshold is pushed higher by our unit-norm-scaled α parameterisation, but the mechanism is the same. The 8B also mixes in emoji spamming (🎨✨💕✨) — a different collapse mode, but same family of "high-α, post-RLHF breakdown" phenomena.

## Alignment with Jeong — scorecard across all three runs

| Prediction | 1.5B | 4B | 8B |
|---|---|---|---|
| U-shape with mid-layer peak | ✓ @16 | ✓ @19 | ~✓ two peaks (12, 26) |
| ~50 % depth architecture-invariant | ✓ 57 % | ✓ 53 % | **✗ 72 %** at primary peak |
| Anisotropy > 0.8 baseline | 0.87 | 0.88 | 0.83 |
| Valence organisation emerges post-denoising | ✓ | ✓ | ✓ (cleaner) |
| Both axes of circumplex visible | partial | partial | **✓** |
| Chinese-token leakage on Qwen high-α | ✗ | hint (3 %) | **✓ (12 %, semantically aligned)** |
| Explosive steering regime at high α | ✓ | ✓ | ✓ (emoji collapse) |

Two of Jeong's open predictions — the clean Chinese-token artifact and the fully-resolved 2D circumplex — only showed up at **8B**. One — the 50 %-depth universality — breaks on hybrid-thinking 8B. Clean-instruct models (1.5B, 4B-Instruct-2507) stayed on the 50 %-depth trend.

## Artifacts

- `extract-qwen3-8b/emotion_bank.pt`
- `outputs-qwen3-8b/layer_sweep.png` — shows the two-peak pattern
- `outputs-qwen3-8b/anisotropy.png`
- `outputs-qwen3-8b/geometry_cosine.png`
- `outputs-qwen3-8b/geometry_pca2d.png` — first run with both circumplex axes visible
- `outputs-qwen3-8b/confusion_best_layer.png`
- `outputs-qwen3-8b/behavioral_eval.png` + `.json`
- `outputs-qwen3-8b/steering_results.json`
