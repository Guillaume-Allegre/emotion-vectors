"""Defaults for the emotion-vectors reproduction."""
from __future__ import annotations
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
RUNS_DIR = PACKAGE_ROOT / "runs"

# 20 emotions grouped by valence × arousal quadrant.
DEFAULT_EMOTIONS: list[str] = [
    "excited", "joyful", "proud", "amused", "enthusiastic",
    "content", "calm", "grateful", "relieved", "affectionate",
    "angry", "afraid", "anxious", "desperate", "disgusted",
    "sad", "ashamed", "guilty", "lonely", "bored",
]

QUADRANT: dict[str, str] = {
    **{e: "HAP" for e in ["excited", "joyful", "proud", "amused", "enthusiastic"]},
    **{e: "LAP" for e in ["content", "calm", "grateful", "relieved", "affectionate"]},
    **{e: "HAN" for e in ["angry", "afraid", "anxious", "desperate", "disgusted"]},
    **{e: "LAN" for e in ["sad", "ashamed", "guilty", "lonely", "bored"]},
}

TOKEN_SKIP = 50
N_STORIES_PER_EMOTION = 30
N_NEUTRAL_STORIES = 200

# The shared Modal volume keeps: /corpus/*.json + /runs/<run_name>/*
MODAL_VOL = "emotion-vectors-vol"
MODAL_VOL_PATH = "/vol"
MODAL_APP_NAME = "emotionvec"

EMOTION_STORIES_NAME = "emotion_stories.json"
NEUTRAL_STORIES_NAME = "neutral_stories.json"
