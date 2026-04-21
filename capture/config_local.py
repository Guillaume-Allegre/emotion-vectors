"""Local config for the hack_interp pipeline."""
from pathlib import Path

# Roots
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent       # /home/user/safet
EXP_ROOT  = THIS_DIR.parent              # /home/user/safet/hack_interp

# Model
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

# Emotion bank (from emotion_vectors repo)
EMOTION_BANK_PATH   = REPO_ROOT / "emotion_vectors" / "qwen3-4b" / "extract-qwen3-4b" / "emotion_bank.pt"
BEST_LAYER          = 19                 # see emotion_bank['best_layer']

# The 20 emotions, in the canonical order the bank was built with.
# Must match emotion_vectors/config.py::EMOTIONS.
EMOTIONS = [
    "excited", "joyful", "proud", "amused", "enthusiastic",
    "content", "calm", "grateful", "relieved", "affectionate",
    "angry", "afraid", "anxious", "desperate", "disgusted",
    "sad", "ashamed", "guilty", "lonely", "bored",
]

# Dataset paths
DATASET_ROOT      = EXP_ROOT / "dataset"
ROLLOUTS_DIR      = DATASET_ROOT / "rollouts"
ACTIVATIONS_DIR   = DATASET_ROOT / "activations"
ROWS_PARQUET      = DATASET_ROOT / "rows.parquet"
MANIFEST_PATH     = DATASET_ROOT / "manifest.json"
NEUTRAL_STATS_PATH = EXP_ROOT / "capture" / "neutral_stats_layer19.json"

# HF cache redirect — keep model weights off the tiny /home volume.
HF_HOME_DEFAULT = Path("/var/tmp/rhb-models")

# Runner defaults
MAX_STEPS = 40
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
