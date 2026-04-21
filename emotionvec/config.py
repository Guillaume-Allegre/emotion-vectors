"""Defaults for the emotion-vectors reproduction."""
from __future__ import annotations
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
RUNS_DIR = PACKAGE_ROOT / "runs"

# 40 emotions grouped by valence × arousal quadrant (10 per quadrant).
_HAP = ["excited", "joyful", "proud", "amused", "enthusiastic",
        "ecstatic", "thrilled", "elated", "triumphant", "inspired"]
_LAP = ["content", "calm", "grateful", "relieved", "affectionate",
        "serene", "peaceful", "nostalgic", "satisfied", "tender"]
_HAN = ["angry", "afraid", "anxious", "desperate", "disgusted",
        "furious", "terrified", "jealous", "outraged", "frustrated"]
_LAN = ["sad", "ashamed", "guilty", "lonely", "bored",
        "melancholic", "regretful", "humiliated", "hopeless", "disappointed"]

DEFAULT_EMOTIONS: list[str] = _HAP + _LAP + _HAN + _LAN

QUADRANT: dict[str, str] = {
    **{e: "HAP" for e in _HAP},
    **{e: "LAP" for e in _LAP},
    **{e: "HAN" for e in _HAN},
    **{e: "LAN" for e in _LAN},
}

# The 20-emotion subset used in the pre-v0.2 runs (still on the volume under
# old run names). Kept for reproducing the legacy result.
LEGACY_EMOTIONS_20: list[str] = [
    "excited", "joyful", "proud", "amused", "enthusiastic",
    "content", "calm", "grateful", "relieved", "affectionate",
    "angry", "afraid", "anxious", "desperate", "disgusted",
    "sad", "ashamed", "guilty", "lonely", "bored",
]

TOKEN_SKIP = 50
N_STORIES_PER_EMOTION = 30
N_NEUTRAL_STORIES = 200

# The shared Modal volume keeps: /corpus/*.json + /runs/<run_name>/*
MODAL_VOL = "emotion-vectors-vol"
MODAL_VOL_PATH = "/vol"
MODAL_APP_NAME = "emotionvec"

EMOTION_STORIES_NAME = "emotion_stories.json"
NEUTRAL_STORIES_NAME = "neutral_stories.json"
