"""Story corpus: load local JSONs, optionally generate via OpenAI.

Corpus format:
  emotion_stories.json : [{"emotion": str, "setting": str, "text": str}, ...]
  neutral_stories.json : [{"topic": str, "text": str}, ...]
"""
from __future__ import annotations
import json
from pathlib import Path

from .config import DATA_DIR, EMOTION_STORIES_NAME, NEUTRAL_STORIES_NAME


def local_emotion_stories() -> Path:
    return DATA_DIR / EMOTION_STORIES_NAME


def local_neutral_stories() -> Path:
    return DATA_DIR / NEUTRAL_STORIES_NAME


def read_corpus(emotion_path: Path, neutral_path: Path) -> tuple[list[dict], list[dict]]:
    emo = json.loads(emotion_path.read_text())
    neu = json.loads(neutral_path.read_text())
    return emo, neu


def validate_corpus_coverage(emotions: list[str], emotion_stories: list[dict]) -> list[str]:
    """Return the list of emotions NOT present in the corpus."""
    present = {rec.get("emotion") for rec in emotion_stories}
    return [e for e in emotions if e not in present]
