"""Load the Qwen3-4B emotion bank and project activations onto it.

The bank at `emotion_vectors/qwen3-4b/extract-qwen3-4b/emotion_bank.pt` has:
    {
      "bank":           {emotion: {layer_idx: tensor(d_model,)}},  # PCA-denoised
      "bank_raw":       same,                                      # mean-sub only
      "best_layer":     19,  # meaning: out.hidden_states[best_layer + 1]
      "pca_components": ...,
      "layer_acc_raw":  np.array(n_layers),
      ...
    }

The extraction code (`emotion_vectors/modal_app_qwen3.py` line 366) uses
`out.hidden_states[layer + 1]` as the activation at the bank's `layer`. This
module uses the same convention.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch

from config_local import EMOTION_BANK_PATH, EMOTIONS, NEUTRAL_STATS_PATH, BEST_LAYER


class EmotionProjector:
    """Project a hidden-state vector onto the 20 emotion directions at layer 19,
    returning a 20-dim z-scored cosine-similarity vector."""

    def __init__(self, bank_path: Path = None, layer: int = None,
                 neutral_stats_path: Path = None):
        bank_path = Path(bank_path or EMOTION_BANK_PATH)
        data = torch.load(bank_path, map_location="cpu", weights_only=False)
        bank = data["bank"]
        self.layer = int(layer if layer is not None else data["best_layer"])
        # stack in canonical EMOTIONS order → (20, d_model)
        vecs = torch.stack([bank[e][self.layer] for e in EMOTIONS]).to(torch.float32)
        # normalize to unit vectors for cosine similarity
        self.V = vecs / vecs.norm(dim=-1, keepdim=True)        # (20, d_model)
        self.emotions = list(EMOTIONS)
        # Neutral baseline: precomputed mean/std of cosine(vec, h_neutral) for z-scoring.
        # If not yet computed, project against zero — scores are raw cosines.
        nstats_path = Path(neutral_stats_path or NEUTRAL_STATS_PATH)
        if nstats_path.exists():
            s = json.loads(nstats_path.read_text())
            self.neutral_mean = torch.tensor(s["mean"], dtype=torch.float32)
            self.neutral_std  = torch.tensor(s["std"],  dtype=torch.float32)
        else:
            self.neutral_mean = torch.zeros(20)
            self.neutral_std  = torch.ones(20)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d_model). Returns (..., 20) z-scored cosine similarities."""
        h32 = h.to(torch.float32)
        h_n = h32 / h32.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cos = h_n @ self.V.T                                   # (..., 20)
        z = (cos - self.neutral_mean) / self.neutral_std.clamp(min=1e-6)
        return z

    def save_neutral_stats(self, neutral_hidden: torch.Tensor, out_path: Path) -> None:
        """Compute neutral mean/std over a batch of neutral activations.

        neutral_hidden: (N, d_model) — a sample of residual-stream activations from
        neutral (non-task) text at layer `self.layer`. We cosine-project them onto
        V, then store the per-emotion mean and std. Subsequent projections z-score
        against these statistics, so `z=+2` means "two sigmas above typical".
        """
        h32 = neutral_hidden.to(torch.float32)
        h_n = h32 / h32.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cos = h_n @ self.V.T                                   # (N, 20)
        mean = cos.mean(dim=0)
        std  = cos.std(dim=0).clamp(min=1e-6)
        out_path.write_text(json.dumps({
            "emotions": self.emotions,
            "layer":    self.layer,
            "mean":     mean.tolist(),
            "std":      std.tolist(),
            "n_samples": int(neutral_hidden.shape[0]),
        }, indent=2))
        self.neutral_mean = mean
        self.neutral_std  = std
