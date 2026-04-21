"""Qwen3-8B served as an OpenAI-compatible endpoint on Modal, with
side-channel activation capture.

Design:
- One A100-40GB Modal container holds the model. `keep_warm=1` keeps it
  loaded across Harbor rollouts.
- FastAPI app exposes `/v1/chat/completions` + `/v1/models` so Harbor's
  `terminus` agent (and anything LiteLLM-routable) just works by pointing
  `OPENAI_API_BASE` at our endpoint.
- Every request gets a UUID. Forward hooks at layers {12,20,26,32}
  capture the residual stream of the generated tokens. We save a .npz
  at `/qwen_acts/<request_id>.npz` to the Modal volume `qwen-acts-vol`.
- The UUID is returned in `response.id` so Harbor's trajectory logs it
  and we can stitch post-hoc.

Deploy:
    modal deploy hack_interp/modal/qwen_server.py
    # prints an HTTPS URL; use that as OPENAI_API_BASE in Harbor.
"""
from __future__ import annotations
import json, os, time, uuid
from pathlib import Path

import modal

APP_NAME = "qwen3-8b-server"
MODEL_NAME = "Qwen/Qwen3-8B"            # hybrid; enable_thinking=False
CAPTURE_LAYERS = [12, 20, 26, 32]       # multi-layer capture
BEST_LAYER = 26                          # the emotion-bank's preferred layer
D_MODEL = 4096                           # Qwen3-8B
GPU = "A100-40GB"

try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    EMOTION_BANK_LOCAL = REPO_ROOT / "emotion_vectors" / "qwen3-8b" / "extract-qwen3-8b" / "emotion_bank.pt"
except IndexError:
    EMOTION_BANK_LOCAL = Path("/placeholder/emotion_bank.pt")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.55.0",
        "accelerate>=1.0",
        "numpy==1.26.4",
        "huggingface_hub>=0.34.0",
        "hf_transfer==0.1.8",
        "fastapi==0.117.0",
        "pydantic==2.9.2",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_file(str(EMOTION_BANK_LOCAL), "/emotion_bank.pt", copy=True)
)

acts_vol  = modal.Volume.from_name("qwen-acts-vol",  create_if_missing=True)
hf_cache  = modal.Volume.from_name("hf-cache",       create_if_missing=True)
ACTS_PATH = "/qwen_acts"
HF_CACHE  = "/root/.cache/huggingface"

app = modal.App(APP_NAME, image=image)


@app.cls(
    gpu=GPU,
    volumes={ACTS_PATH: acts_vol, HF_CACHE: hf_cache},
    timeout=3600,
    max_containers=4,
    min_containers=1,
    scaledown_window=600,   # keep warm ~10 min after last request
)
class QwenServer:

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[load] {MODEL_NAME}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="cuda",
            attn_implementation="sdpa",
        )
        self.model.eval()

        # Emotion bank at layer BEST_LAYER; unit-normalised on CPU.
        bank = torch.load("/emotion_bank.pt", map_location="cpu", weights_only=False)
        emotions = [
            "excited","joyful","proud","amused","enthusiastic",
            "content","calm","grateful","relieved","affectionate",
            "angry","afraid","anxious","desperate","disgusted",
            "sad","ashamed","guilty","lonely","bored",
        ]
        V = torch.stack([bank["bank"][e][BEST_LAYER] for e in emotions]).to(torch.float32)
        self.V = (V / V.norm(dim=-1, keepdim=True))
        self.emotions = emotions

        # Forward hooks on the capture layers. Each layer's hook fires once
        # per forward pass (prompt prefill + one per decode step during
        # generate). We clear the buffer before each request.
        self.torch = torch
        self._acts: dict[int, list] = {L: [] for L in CAPTURE_LAYERS}

        def _make_hook(L):
            def hook(mod, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                self._acts[L].append(h.detach().to(torch.float16).cpu())
            return hook

        self._hook_handles = []
        for L in CAPTURE_LAYERS:
            target = self.model.model.layers[L]
            self._hook_handles.append(target.register_forward_hook(_make_hook(L)))
        print(f"[ready] hooks on layers {CAPTURE_LAYERS}", flush=True)

    def _clear_acts(self):
        for L in CAPTURE_LAYERS:
            self._acts[L].clear()

    def _collect(self, n_gen: int, prompt_len: int) -> dict:
        """Build per-layer tensors from hook buffers after one generate() call.
        Hook captures: acts[0] is prefill (1, prompt_len, d), acts[1..n_gen-1]
        are per-decode-token (1, 1, d).
        Decision-activation for generated token i is:
          i == 0           → last position of prefill pass
          i >= 1           → position 0 of acts[i]
        """
        import numpy as np, torch
        out = {}
        for L in CAPTURE_LAYERS:
            acts = self._acts[L]
            if not acts:
                continue
            decisions = [acts[0][0, -1:, :]]
            for a in acts[1:max(1, n_gen)]:
                decisions.append(a[0])
            per_tok = torch.cat(decisions, dim=0).to(torch.float32)  # (n_gen, d)
            last_h = per_tok[-1].to(torch.float16).numpy()
            mean_h = per_tok.mean(dim=0).to(torch.float16).numpy()
            per_h16 = per_tok.to(torch.float16).numpy()
            out[f"layer_{L}_last_h"]      = last_h
            out[f"layer_{L}_mean_h"]      = mean_h
            out[f"layer_{L}_per_token_h"] = per_h16
            if L == BEST_LAYER:
                h_n = per_tok.mean(dim=0) / per_tok.mean(dim=0).norm().clamp(min=1e-6)
                cos = (h_n @ self.V.T).to(torch.float32).numpy()
                out["emotion_z_best_layer"] = cos
        out["n_gen_tokens"] = int(n_gen)
        out["prompt_len"]    = int(prompt_len)
        out["capture_layers"] = CAPTURE_LAYERS
        out["best_layer"]     = BEST_LAYER
        out["d_model"]        = D_MODEL
        return out

    @modal.asgi_app()
    def fastapi_app(self):
        from fastapi import FastAPI, HTTPException
        import numpy as np, torch

        web = FastAPI()

        @web.get("/v1/models")
        def list_models():
            return {"object": "list", "data": [
                {"id": "qwen3-8b", "object": "model", "owned_by": "hack-interp"}]}

        @web.post("/v1/chat/completions")
        async def chat_completions(body: dict):
            messages     = body.get("messages") or []
            max_tokens   = int(body.get("max_tokens", body.get("max_new_tokens", 16000)))
            temperature  = float(body.get("temperature", 0.7))
            stop         = body.get("stop", None)        # list[str] or None

            req_id = "qwen-" + uuid.uuid4().hex[:16]
            t0 = time.time()

            # Tokenize
            prompt_text = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
            prompt_ids = self.tok(prompt_text, return_tensors="pt").input_ids.to("cuda")
            prompt_len = prompt_ids.shape[1]

            # Generate with hooks active
            self._clear_acts()
            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=self.tok.eos_token_id,
                return_dict_in_generate=True,
            )
            try:
                with torch.no_grad():
                    out = self.model.generate(prompt_ids, **gen_kwargs)
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                raise HTTPException(status_code=507, detail=f"CUDA OOM: {e}")
            gen_ids = out.sequences[0, prompt_len:]
            text = self.tok.decode(gen_ids, skip_special_tokens=True)
            # Trim at stop strings if provided (OpenAI semantics)
            if stop:
                stops = stop if isinstance(stop, list) else [stop]
                for s in stops:
                    idx = text.find(s)
                    if idx >= 0:
                        text = text[:idx]
                        break
            n_gen = int(gen_ids.shape[0])

            # Collect per-layer activations and persist.
            try:
                acts_dict = self._collect(n_gen, prompt_len)
                out_path = Path(ACTS_PATH) / f"{req_id}.npz"
                np.savez_compressed(out_path, **{
                    k: v for k, v in acts_dict.items()
                    if isinstance(v, np.ndarray) or isinstance(v, (int, float, list))
                })
                acts_vol.commit()
            except Exception as e:
                print(f"[warn] activation save failed for {req_id}: {e}", flush=True)
            finally:
                torch.cuda.empty_cache()

            dt = time.time() - t0
            finish_reason = "stop" if n_gen < max_tokens else "length"

            return {
                "id":      req_id,
                "object":  "chat.completion",
                "created": int(time.time()),
                "model":   "qwen3-8b",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens":     prompt_len,
                    "completion_tokens": n_gen,
                    "total_tokens":      prompt_len + n_gen,
                },
                "x_rhb": {
                    "request_id":    req_id,
                    "wallclock_sec": round(dt, 3),
                    "activation_path": str(out_path) if 'out_path' in locals() else None,
                },
            }

        @web.get("/healthz")
        def health():
            return {"ok": True, "model": MODEL_NAME,
                    "layers_captured": CAPTURE_LAYERS, "best_layer": BEST_LAYER}

        return web
