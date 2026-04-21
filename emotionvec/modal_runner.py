"""Unified Modal app for emotion-vector extraction.

One app, parameterised by (model, emotions, run name). Produces per-run
artefacts on the Modal volume `emotion-vectors-vol`:

    /vol/corpus/{emotion_stories,neutral_stories}.json   # shared
    /vol/runs/<run_name>/activations.pt                  # per-story × layer × d
    /vol/runs/<run_name>/neutral_activations.pt
    /vol/runs/<run_name>/emotion_bank.pt                 # bank + full metadata
    /vol/runs/<run_name>/layer_selection.json            # JSON: best layer, acc
    /vol/runs/<run_name>/*.png                           # diagnostic plots
    /vol/runs/<run_name>/steering_results.json           # optional
    /vol/runs/<run_name>/behavioral_eval.json            # optional

Invoke via the `emotionvec` CLI — `python -m emotionvec run ...` translates
into the appropriate `modal run` call.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import modal

from .config import (
    DEFAULT_EMOTIONS,
    QUADRANT,
    TOKEN_SKIP as DEFAULT_TOKEN_SKIP,
    MODAL_APP_NAME,
    MODAL_VOL,
    MODAL_VOL_PATH,
    EMOTION_STORIES_NAME,
    NEUTRAL_STORIES_NAME,
)


# ---- Modal image ----
# torch 2.6 + transformers 4.55 cover every Qwen line we've targeted
# (Qwen2.5, Qwen3, Qwen3-hybrid-thinking). Llama-3 / Mistral also work.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.55.0",
        "accelerate>=1.0",
        "numpy==1.26.4",
        "scikit-learn==1.5.2",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "tqdm==4.66.5",
        "huggingface_hub>=0.34.0",
        "hf_transfer==0.1.8",
        "sentencepiece==0.2.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name(MODAL_VOL, create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
HF_CACHE = "/root/.cache/huggingface"

app = modal.App(MODAL_APP_NAME, image=image)

CORPUS_DIR = "corpus"
RUNS_DIR = "runs"


def _run_path(run_name: str) -> Path:
    return Path(MODAL_VOL_PATH) / RUNS_DIR / run_name


def _parse_emotions(emotions: list[str] | str | None) -> list[str]:
    if emotions is None:
        return list(DEFAULT_EMOTIONS)
    if isinstance(emotions, str):
        if emotions in ("default", "", "all"):
            return list(DEFAULT_EMOTIONS)
        return [e.strip() for e in emotions.split(",") if e.strip()]
    return list(emotions)


# ---- volume helpers ----
@app.function(volumes={MODAL_VOL_PATH: vol}, timeout=300)
def put_file(name: str, data: bytes) -> str:
    p = Path(MODAL_VOL_PATH) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    vol.commit()
    return str(p)


@app.function(volumes={MODAL_VOL_PATH: vol}, timeout=300)
def get_file(name: str) -> bytes:
    return (Path(MODAL_VOL_PATH) / name).read_bytes()


@app.function(volumes={MODAL_VOL_PATH: vol}, timeout=60)
def list_vol(prefix: str = "") -> list[str]:
    root = Path(MODAL_VOL_PATH)
    start = root / prefix if prefix else root
    if not start.exists():
        return []
    return [str(p.relative_to(root)) for p in start.rglob("*") if p.is_file()]


# ---- activation extraction ----
@app.function(
    gpu="A100-40GB",
    volumes={MODAL_VOL_PATH: vol, HF_CACHE: hf_cache},
    timeout=60 * 120,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": os.environ.get("HF_TOKEN", "")})],
)
def extract_all(
    run_name: str,
    model_name: str,
    emotions: list[str] | str | None = None,
    token_skip: int = DEFAULT_TOKEN_SKIP,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    emotions = _parse_emotions(emotions)
    run_dir = _run_path(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] run={run_name} model={model_name} n_emotions={len(emotions)}", flush=True)

    print(f"[extract] loading {model_name}…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"[extract] n_layers={n_layers} d_model={d_model}", flush=True)

    corpus_root = Path(MODAL_VOL_PATH) / CORPUS_DIR
    emo = json.loads((corpus_root / EMOTION_STORIES_NAME).read_text())
    neu = json.loads((corpus_root / NEUTRAL_STORIES_NAME).read_text())

    # Restrict to requested emotions.
    emo = [r for r in emo if r.get("emotion") in set(emotions)]
    present = sorted({r["emotion"] for r in emo})
    missing = sorted(set(emotions) - set(present))
    if missing:
        raise RuntimeError(
            f"Corpus missing stories for: {missing}. "
            f"Regenerate corpus (emotionvec gen-corpus) or drop these."
        )
    print(f"[extract] corpus: {len(emo)} emotion stories (over {len(present)}) / "
          f"{len(neu)} neutral stories", flush=True)

    @torch.no_grad()
    def get_acts(text: str):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to("cuda")
        out = model(ids, output_hidden_states=True)
        hs = torch.stack(out.hidden_states[1:], dim=0).squeeze(1)
        if hs.shape[1] <= token_skip:
            return hs.mean(dim=1).to(torch.float32).cpu()
        return hs[:, token_skip:, :].mean(dim=1).to(torch.float32).cpu()

    emo_acts = torch.zeros(len(emo), n_layers, d_model, dtype=torch.float32)
    emo_labels = []
    for i, rec in enumerate(tqdm(emo, desc="emotion")):
        emo_acts[i] = get_acts(rec["text"])
        emo_labels.append(rec["emotion"])

    neu_acts = torch.zeros(len(neu), n_layers, d_model, dtype=torch.float32)
    for i, rec in enumerate(tqdm(neu, desc="neutral")):
        neu_acts[i] = get_acts(rec["text"])

    torch.save(
        {"activations": emo_acts, "labels": emo_labels,
         "d_model": d_model, "n_layers": n_layers,
         "model_name": model_name, "token_skip": token_skip,
         "emotions": emotions},
        run_dir / "activations.pt",
    )
    torch.save(
        {"activations": neu_acts, "d_model": d_model, "n_layers": n_layers,
         "model_name": model_name},
        run_dir / "neutral_activations.pt",
    )
    vol.commit()
    print(f"[extract] done -> {run_dir}", flush=True)
    return {
        "run_name": run_name, "n_emotion_stories": len(emo),
        "n_neutral_stories": len(neu),
        "n_layers": n_layers, "d_model": d_model,
    }


# ---- vectors + layer selection ----
@app.function(
    gpu=None, cpu=4, volumes={MODAL_VOL_PATH: vol}, timeout=60 * 45,
)
def build_vectors_and_validate(
    run_name: str,
    emotions: list[str] | str | None = None,
) -> dict:
    import random
    import numpy as np
    import torch
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    emotions = _parse_emotions(emotions)
    run_dir = _run_path(run_name)
    emo = torch.load(run_dir / "activations.pt", map_location="cpu", weights_only=False)
    neu = torch.load(run_dir / "neutral_activations.pt", map_location="cpu", weights_only=False)
    model_name = emo.get("model_name", "unknown")

    A = emo["activations"]
    labels = np.array(emo["labels"])
    NEU = neu["activations"]
    N, L, d = A.shape
    print(f"[build] run={run_name} model={model_name} N={N} L={L} d={d} M={NEU.shape[0]}", flush=True)

    # 70/30 stratified split per emotion.
    rng = random.Random(0)
    train_idx, test_idx = [], []
    for e in emotions:
        idx = np.where(labels == e)[0].tolist()
        rng.shuffle(idx)
        k = int(0.7 * len(idx))
        train_idx.extend(idx[:k])
        test_idx.extend(idx[k:])
    train_idx = np.array(train_idx); test_idx = np.array(test_idx)

    A_train = A[train_idx]; L_train = labels[train_idx]
    A_test = A[test_idx];  L_test = labels[test_idx]

    bank_raw: dict[str, dict[int, torch.Tensor]] = {e: {} for e in emotions}
    bank: dict[str, dict[int, torch.Tensor]] = {e: {} for e in emotions}
    pca_components: dict[int, torch.Tensor] = {}
    anisotropy = np.zeros(L)
    layer_acc_raw = np.zeros(L)
    layer_acc_denoise = np.zeros(L)

    for layer in range(L):
        mu_all = A_train[:, layer, :].mean(dim=0)
        for e in emotions:
            mask = (L_train == e)
            mu_e = A_train[mask, layer, :].mean(dim=0)
            bank_raw[e][layer] = (mu_e - mu_all).clone()

        NL = NEU[:, layer, :].numpy()
        pca = PCA(n_components=min(NL.shape) - 1).fit(NL)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cum, 0.5)) + 1
        P = torch.tensor(pca.components_[:k], dtype=torch.float32)
        pca_components[layer] = P

        for e in emotions:
            v = bank_raw[e][layer]
            bank[e][layer] = v - (v @ P.T) @ P

        sample = NEU[:min(100, NEU.shape[0]), layer, :]
        sn = sample / sample.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sim = sn @ sn.T
        iu = torch.triu_indices(sim.shape[0], sim.shape[1], offset=1)
        anisotropy[layer] = sim[iu[0], iu[1]].mean().item()

        for bank_use, out_arr in [(bank_raw, layer_acc_raw), (bank, layer_acc_denoise)]:
            V = torch.stack([bank_use[e][layer] for e in emotions])
            Vn = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            H = A_test[:, layer, :]
            Hn = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            pred_idx = (Hn @ Vn.T).argmax(dim=-1).numpy()
            pred = np.array([emotions[i] for i in pred_idx])
            out_arr[layer] = float((pred == L_test).mean())

        if layer % 4 == 0:
            print(f"  layer {layer:>3}: acc_raw={layer_acc_raw[layer]:.3f} "
                  f"acc_denoise={layer_acc_denoise[layer]:.3f} "
                  f"aniso={anisotropy[layer]:.3f} k_pca={k}", flush=True)

    best_layer = int(np.argmax(layer_acc_denoise))
    best_acc = float(layer_acc_denoise[best_layer])
    print(f"[build] BEST layer (denoised): {best_layer}  acc={best_acc:.3f}", flush=True)

    # Neutral z-score reference for probe/analyze later.
    neutral_mean_per_layer = []
    neutral_std_per_layer = []
    for layer in range(L):
        V = torch.stack([bank[e][layer] for e in emotions])
        Vn = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        NLn = NEU[:, layer, :] / NEU[:, layer, :].norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scores = (NLn @ Vn.T).numpy()
        neutral_mean_per_layer.append(scores.mean(axis=0).tolist())
        neutral_std_per_layer.append(scores.std(axis=0).clip(min=1e-6).tolist())

    torch.save(
        {
            "bank": bank,
            "bank_raw": bank_raw,
            "pca_components": pca_components,
            "best_layer": best_layer,
            "layer_acc_raw": layer_acc_raw.tolist(),
            "layer_acc_denoise": layer_acc_denoise.tolist(),
            "anisotropy": anisotropy.tolist(),
            "emotions": emotions,
            "model_name": model_name,
            "d_model": d, "n_layers": L,
            "train_idx": train_idx.tolist(), "test_idx": test_idx.tolist(),
            "neutral_mean_per_layer": neutral_mean_per_layer,
            "neutral_std_per_layer": neutral_std_per_layer,
        },
        run_dir / "emotion_bank.pt",
    )

    # JSON summary — the "load me later" contract.
    (run_dir / "layer_selection.json").write_text(json.dumps({
        "model_name": model_name,
        "emotions": emotions,
        "best_layer": best_layer,
        "best_acc_denoised": best_acc,
        "best_acc_raw": float(layer_acc_raw[best_layer]),
        "n_layers": L,
        "d_model": d,
        "layer_acc_raw": layer_acc_raw.tolist(),
        "layer_acc_denoise": layer_acc_denoise.tolist(),
        "anisotropy": anisotropy.tolist(),
        "mean_anisotropy": float(anisotropy.mean()),
    }, indent=2))

    # Plots. Quadrant colouring is best-effort — if a custom emotion isn't in
    # the 4-quadrant dictionary it falls back to grey.
    _render_plots(run_dir, model_name, emotions, bank, A_test, L_test,
                  best_layer, layer_acc_raw, layer_acc_denoise, anisotropy)

    vol.commit()
    return {
        "run_name": run_name, "model_name": model_name,
        "best_layer": best_layer, "best_acc_denoised": best_acc,
        "best_acc_raw": float(layer_acc_raw[best_layer]),
        "mean_anisotropy": float(anisotropy.mean()),
        "anisotropy_at_best": float(anisotropy[best_layer]),
        "n_layers": L, "d_model": d,
    }


def _render_plots(run_dir, model_name, emotions, bank, A_test, L_test,
                  best_layer, layer_acc_raw, layer_acc_denoise, anisotropy):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA as SkPCA

    L = len(layer_acc_raw)

    # Layer sweep
    plt.figure(figsize=(8, 4))
    plt.plot(range(L), layer_acc_raw, label="raw (mean-sub)")
    plt.plot(range(L), layer_acc_denoise, label="PCA-denoised")
    plt.axhline(1.0 / len(emotions), color="gray", ls="--",
                label=f"chance (1/{len(emotions)})")
    plt.axvline(best_layer, color="red", ls=":", alpha=0.5, label=f"best={best_layer}")
    plt.xlabel("layer"); plt.ylabel(f"{len(emotions)}-way held-out accuracy")
    plt.title(f"Classification accuracy by layer — {model_name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / "layer_sweep.png", dpi=120); plt.close()

    # Anisotropy
    plt.figure(figsize=(8, 3.5))
    plt.plot(range(L), anisotropy)
    plt.xlabel("layer"); plt.ylabel("mean pairwise cos sim (neutral)")
    plt.title(f"Anisotropy by layer — {model_name}")
    plt.tight_layout()
    plt.savefig(run_dir / "anisotropy.png", dpi=120); plt.close()

    # Geometry cosine at best layer (quadrant-ordered if possible).
    def quad_order(es):
        preferred = ["HAP", "LAP", "LAN", "HAN"]
        return sorted(es, key=lambda e: (
            preferred.index(QUADRANT[e]) if e in QUADRANT else len(preferred),
            e,
        ))

    order = quad_order(emotions)
    Vs = torch.stack([bank[e][best_layer] for e in order])
    Vsn = Vs / Vs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = (Vsn @ Vsn.T).numpy()
    plt.figure(figsize=(max(9, 0.4 * len(order)), max(8, 0.38 * len(order))))
    sns.heatmap(sim, xticklabels=order, yticklabels=order, cmap="coolwarm",
                vmin=-1, vmax=1, square=True, cbar_kws={"label": "cosine"})
    plt.title(f"Emotion vector cosine (layer {best_layer}) — {model_name}")
    plt.tight_layout()
    plt.savefig(run_dir / "geometry_cosine.png", dpi=120); plt.close()

    # 2D PCA
    V_all = torch.stack([bank[e][best_layer] for e in emotions]).numpy()
    pc2 = SkPCA(n_components=2).fit_transform(V_all)
    colors = {"HAP": "#d62728", "LAP": "#2ca02c", "HAN": "#9467bd", "LAN": "#1f77b4"}
    plt.figure(figsize=(8, 6))
    for i, e in enumerate(emotions):
        c = colors.get(QUADRANT.get(e, ""), "#888888")
        plt.scatter(pc2[i, 0], pc2[i, 1], c=c, s=60)
        plt.annotate(e, (pc2[i, 0], pc2[i, 1]), fontsize=9,
                     xytext=(5, 3), textcoords="offset points")
    for q, c in colors.items():
        plt.scatter([], [], c=c, label=q)
    plt.legend(title="quadrant")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"2D PCA (layer {best_layer}) — {model_name}")
    plt.tight_layout()
    plt.savefig(run_dir / "geometry_pca2d.png", dpi=120); plt.close()

    # Confusion at best layer
    V = torch.stack([bank[e][best_layer] for e in emotions])
    Vn = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    H = A_test[:, best_layer, :]
    Hn = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    preds = (Hn @ Vn.T).argmax(dim=-1).numpy()
    cm = np.zeros((len(emotions), len(emotions)))
    for true_e, pi in zip(L_test, preds):
        cm[emotions.index(true_e), pi] += 1
    cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)
    plt.figure(figsize=(max(9, 0.4 * len(emotions)), max(8, 0.38 * len(emotions))))
    sns.heatmap(cm, xticklabels=emotions, yticklabels=emotions, cmap="magma",
                vmin=0, vmax=1, square=True, cbar_kws={"label": "row-normalised"})
    plt.title(f"Confusion at layer {best_layer} — {model_name}")
    plt.xlabel("predicted"); plt.ylabel("true")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_best_layer.png", dpi=120); plt.close()


# ---- text analysis (probe + arc + heatmap) ----
@app.function(
    gpu="A100-40GB",
    volumes={MODAL_VOL_PATH: vol, HF_CACHE: hf_cache},
    timeout=60 * 20,
)
def analyze_text(
    run_name: str,
    model_name: str,
    text: str,
    layer: int | None = None,
    top_k: int = 3,
    window: int = 16,
    make_plot: bool = True,
) -> dict:
    import hashlib
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = _run_path(run_name)
    bank_data = torch.load(run_dir / "emotion_bank.pt", map_location="cpu", weights_only=False)
    bank = bank_data["bank"]
    emotions = bank_data["emotions"]
    if layer is None:
        layer = int(bank_data["best_layer"])

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    V = torch.stack([bank[e][layer] for e in emotions]).to(torch.float32)
    Vn = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    neu_mean = np.asarray(bank_data["neutral_mean_per_layer"][layer])
    neu_std = np.asarray(bank_data["neutral_std_per_layer"][layer])

    ids = tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids.to("cuda")
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    h = out.hidden_states[layer + 1].squeeze(0).to(torch.float32).cpu()
    hn = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    raw = (hn @ Vn.T).numpy()
    z = (raw - neu_mean) / neu_std

    toks = tok.convert_ids_to_tokens(ids[0])
    display_toks = [t.replace("Ġ", " ").replace("Ċ", "\\n") for t in toks]
    seq = z.shape[0]

    mean_z = z.mean(axis=0)
    order = np.argsort(-mean_z)
    overall = [(emotions[i], float(mean_z[i])) for i in order]

    per_token = []
    for t in range(seq):
        tops = np.argsort(-z[t])[:top_k]
        per_token.append({"token": display_toks[t],
                          "top": [(emotions[i], float(z[t][i])) for i in tops]})

    arc = []
    for start in range(0, seq, window):
        end = min(start + window, seq)
        mz = z[start:end].mean(axis=0)
        top_i = int(mz.argmax())
        arc.append({
            "range": [start, end],
            "tokens_preview": "".join(display_toks[start:end])[:80],
            "top_emotion": emotions[top_i], "z": float(mz[top_i]),
            "top_k": [(emotions[i], float(mz[i])) for i in np.argsort(-mz)[:top_k]],
        })

    heatmap_path = None
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        plot_seq = min(120, seq)
        zc = z[:plot_seq]
        plt.figure(figsize=(max(10, plot_seq * 0.22), max(6, 0.35 * len(emotions))))
        sns.heatmap(zc.T, xticklabels=display_toks[:plot_seq], yticklabels=emotions,
                    cmap="coolwarm", center=0, vmin=-3, vmax=3,
                    cbar_kws={"label": "z-score vs neutral"})
        plt.xticks(rotation=90, fontsize=7)
        plt.title(f"Emotion probe (layer {layer}) — {model_name}")
        plt.tight_layout()
        tag = hashlib.md5(text.encode()).hexdigest()[:8]
        outp = run_dir / f"analyze_{tag}.png"
        plt.savefig(outp, dpi=130); plt.close()
        vol.commit()
        heatmap_path = str(outp.relative_to(Path(MODAL_VOL_PATH)))

    return {
        "run_name": run_name, "model_name": model_name,
        "layer": layer, "n_tokens": seq,
        "overall": overall, "top_k_overall": overall[:top_k],
        "per_token": per_token, "arc": arc,
        "heatmap_path": heatmap_path,
    }


@app.function(
    gpu="A100-40GB",
    volumes={MODAL_VOL_PATH: vol, HF_CACHE: hf_cache},
    timeout=60 * 30,
)
def generate_and_analyze(
    run_name: str,
    model_name: str,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    layer: int | None = None,
    enable_thinking: bool = False,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    kw = {"add_generation_prompt": True, "return_tensors": "pt"}
    try:
        ids = tok.apply_chat_template(messages, enable_thinking=enable_thinking, **kw)
    except TypeError:
        ids = tok.apply_chat_template(messages, **kw)
    ids = ids.to("cuda")
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0, ids.shape[1]:]
    response = tok.decode(gen, skip_special_tokens=True)

    result = analyze_text.local(run_name, model_name, response, layer=layer)
    result["prompt"] = prompt
    result["response"] = response
    return result


# ---- steering sweep ----
@app.function(
    gpu="A100-40GB",
    volumes={MODAL_VOL_PATH: vol, HF_CACHE: hf_cache},
    timeout=60 * 90,
)
def steering_sweep(
    run_name: str,
    model_name: str,
    emotions: list[str] | str | None = None,
    alphas: list[float] | None = None,
    prompts: list[str] | None = None,
    layer: int | None = None,
    max_new_tokens: int = 120,
) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if prompts is None:
        prompts = [
            "Write a short first-person paragraph describing a walk through a park on a Saturday morning.",
            "Write a short first-person paragraph about meeting an old friend at a coffee shop.",
            "Write a short first-person paragraph about sitting on a bus looking out the window.",
            "Write a short first-person paragraph about arriving home after work.",
            "Write a short first-person paragraph about cooking dinner.",
        ]
    if alphas is None:
        alphas = [-8, -4, -2, 0, 2, 4, 8, 12]
    emotions = _parse_emotions(emotions) if emotions else None

    run_dir = _run_path(run_name)
    bank_data = torch.load(run_dir / "emotion_bank.pt", map_location="cpu", weights_only=False)
    bank = bank_data["bank"]
    if layer is None:
        layer = int(bank_data["best_layer"])
    if emotions is None:
        emotions = list(bank.keys())

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()

    neu = torch.load(run_dir / "neutral_activations.pt", map_location="cpu", weights_only=False)
    typical_norm = float(neu["activations"][:, layer, :].norm(dim=-1).mean())
    print(f"[steer] layer={layer} typical_norm={typical_norm:.2f}", flush=True)

    def make_hook(vec):
        v = vec.to(torch.float16).to("cuda")
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                output[0].add_(v)
                return output
            output.add_(v)
            return output
        return hook

    def generate(prompt, hook=None, seed=0):
        messages = [{"role": "user", "content": prompt}]
        kw = {"add_generation_prompt": True, "return_tensors": "pt"}
        try:
            ids = tok.apply_chat_template(messages, enable_thinking=False, **kw)
        except TypeError:
            ids = tok.apply_chat_template(messages, **kw)
        ids = ids.to("cuda")
        handle = model.model.layers[layer].register_forward_hook(hook) if hook else None
        try:
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model.generate(
                    ids, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=0.8, top_p=0.95,
                    pad_token_id=tok.eos_token_id,
                )
        finally:
            if handle is not None:
                handle.remove()
        gen = out[0, ids.shape[1]:]
        return tok.decode(gen, skip_special_tokens=True)

    results = []
    for e in emotions:
        vec = bank[e][layer].clone()
        unit = vec / vec.norm().clamp_min(1e-8)
        for alpha in alphas:
            steer_vec = (alpha * typical_norm * 0.1) * unit
            hook = None if alpha == 0 else make_hook(steer_vec)
            for p_idx, prompt in enumerate(prompts):
                comp = generate(prompt, hook=hook, seed=1000 + p_idx)
                non_ascii = sum(1 for c in comp if ord(c) > 127) / max(len(comp), 1)
                results.append({
                    "emotion": e, "alpha": alpha, "prompt": prompt,
                    "completion": comp, "non_ascii_frac": non_ascii,
                    "layer": layer, "steer_vec_norm": float(steer_vec.norm()),
                })

    outp = run_dir / "steering_results.json"
    outp.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    vol.commit()
    return {"n_results": len(results), "path": str(outp), "layer": layer,
            "typical_norm": typical_norm}


# ---- behavioural classifier eval ----
@app.function(
    gpu="A100-40GB",
    volumes={MODAL_VOL_PATH: vol, HF_CACHE: hf_cache},
    timeout=60 * 30,
)
def behavioral_eval(run_name: str) -> dict:
    from collections import defaultdict
    import numpy as np
    from transformers import pipeline
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = _run_path(run_name)
    results = json.loads((run_dir / "steering_results.json").read_text())
    clf = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None, device=0,
    )

    # j-hartmann classifier supports: anger, disgust, fear, joy, neutral, sadness, surprise
    map_to_clf = {
        "joyful": "joy", "excited": "joy", "amused": "joy", "proud": "joy",
        "enthusiastic": "joy", "grateful": "joy", "affectionate": "joy",
        "content": "joy", "calm": "neutral", "relieved": "joy",
        "angry": "anger", "disgusted": "disgust", "afraid": "fear",
        "anxious": "fear", "desperate": "fear",
        "sad": "sadness", "ashamed": "sadness", "guilty": "sadness",
        "lonely": "sadness", "bored": "neutral",
    }

    buckets = defaultdict(list)
    for r in results:
        buckets[(r["emotion"], r["alpha"])].append(r["completion"])

    report = []
    for (emo, alpha), comps in sorted(buckets.items()):
        target_class = map_to_clf.get(emo, "neutral")
        scores = clf(comps)
        target_scores = []
        class_probs = defaultdict(list)
        for s in scores:
            d = {x["label"].lower(): x["score"] for x in s}
            target_scores.append(d.get(target_class, 0.0))
            for k, v in d.items():
                class_probs[k].append(v)
        report.append({
            "emotion": emo, "alpha": alpha, "target_class": target_class,
            "n": len(comps),
            "target_prob_mean": float(np.mean(target_scores)),
            "target_prob_median": float(np.median(target_scores)),
            "class_probs_mean": {k: float(np.mean(v)) for k, v in class_probs.items()},
            "mean_non_ascii": float(np.mean([x["non_ascii_frac"] for x in results
                                              if x["emotion"] == emo and x["alpha"] == alpha])),
        })

    outp = run_dir / "behavioral_eval.json"
    outp.write_text(json.dumps(report, indent=1, ensure_ascii=False))

    emos = sorted({r["emotion"] for r in report})
    alphas = sorted({r["alpha"] for r in report})
    plt.figure(figsize=(9, 5))
    for emo in emos:
        ys = [next(rr["target_prob_mean"] for rr in report
                   if rr["emotion"] == emo and rr["alpha"] == a) for a in alphas]
        plt.plot(alphas, ys, marker="o", label=emo)
    plt.xlabel("α (steering strength)")
    plt.ylabel("P(target emotion class) — mean over prompts")
    plt.title(f"Behavioural effect of steering — run={run_name}")
    plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / "behavioral_eval.png", dpi=130); plt.close()
    vol.commit()
    return {"report_path": str(outp), "n_buckets": len(report)}


# ---- corpus upload (seed /vol/corpus/ from local data/) ----
@app.function(volumes={MODAL_VOL_PATH: vol}, timeout=300)
def upload_corpus(emotion_json: bytes, neutral_json: bytes) -> dict:
    corpus = Path(MODAL_VOL_PATH) / CORPUS_DIR
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / EMOTION_STORIES_NAME).write_bytes(emotion_json)
    (corpus / NEUTRAL_STORIES_NAME).write_bytes(neutral_json)
    vol.commit()
    return {"emotion_stories": len(json.loads(emotion_json)),
            "neutral_stories": len(json.loads(neutral_json))}


# ---- local CLI entrypoints (used by `emotionvec` CLI) ----
@app.local_entrypoint()
def cli_extract(run_name: str, model_name: str, emotions: str = "default",
                token_skip: int = DEFAULT_TOKEN_SKIP):
    print(extract_all.remote(run_name, model_name, emotions, token_skip))


@app.local_entrypoint()
def cli_build(run_name: str, emotions: str = "default"):
    print(build_vectors_and_validate.remote(run_name, emotions))


@app.local_entrypoint()
def cli_run(run_name: str, model_name: str, emotions: str = "default",
            token_skip: int = DEFAULT_TOKEN_SKIP):
    """Full pipeline: extract → build."""
    print(extract_all.remote(run_name, model_name, emotions, token_skip))
    print(build_vectors_and_validate.remote(run_name, emotions))


@app.local_entrypoint()
def cli_probe(run_name: str, model_name: str, text: str, layer: int = -1):
    lay = None if layer < 0 else layer
    import sys
    result = analyze_text.remote(run_name, model_name, text, lay)
    sys.stdout.write("<<BEGIN_JSON>>")
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.write("<<END_JSON>>\n")


@app.local_entrypoint()
def cli_generate_probe(run_name: str, model_name: str, prompt: str,
                       layer: int = -1, max_new_tokens: int = 200):
    lay = None if layer < 0 else layer
    import sys
    result = generate_and_analyze.remote(run_name, model_name, prompt,
                                         layer=lay, max_new_tokens=max_new_tokens)
    sys.stdout.write("<<BEGIN_JSON>>")
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.write("<<END_JSON>>\n")


@app.local_entrypoint()
def cli_steer(run_name: str, model_name: str, emotions: str = "",
              alphas: str = "-8,-4,-2,0,2,4,8,12", layer: int = -1,
              max_new_tokens: int = 120):
    lay = None if layer < 0 else layer
    alpha_list = [float(x) for x in alphas.split(",") if x.strip()]
    emos = emotions if emotions else None
    print(steering_sweep.remote(run_name, model_name, emos, alpha_list,
                                None, lay, max_new_tokens))


@app.local_entrypoint()
def cli_behav(run_name: str):
    print(behavioral_eval.remote(run_name))


@app.local_entrypoint()
def cli_upload_corpus(emotion_path: str, neutral_path: str):
    emo_bytes = Path(emotion_path).read_bytes()
    neu_bytes = Path(neutral_path).read_bytes()
    print(upload_corpus.remote(emo_bytes, neu_bytes))


@app.local_entrypoint()
def cli_list_vol(prefix: str = ""):
    for p in list_vol.remote(prefix):
        print(p)
