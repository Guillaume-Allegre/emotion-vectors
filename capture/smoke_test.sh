#!/bin/bash
# Shakedown: one 3-step rollout on ticket-backlog-medium, verify the pipeline.
set -e
cd "$(dirname "$0")/.."
source /var/tmp/rhb-venv/bin/activate
export HF_HOME=/var/tmp/rhb-models
TASK=/home/user/safet/envs/reward_hacking_bench/tasks/ticket-backlog-medium
OUT=dataset/rollouts/smoke-qwen3-4b__ticket-backlog-medium__seed0
rm -rf "$OUT"
python capture/run_sml_hf.py "$TASK" --seed 0 --out "$OUT" --max-steps 3 --max-new-tokens 128
echo "---"
ls -la "$OUT"/activations/ "$OUT"/verifier/ 2>&1 || true
python -c "
import numpy as np
from pathlib import Path
for p in sorted(Path('$OUT/activations').glob('*.npz')):
    d = np.load(p)
    print(p.name, {k: d[k].shape if hasattr(d[k], 'shape') else d[k] for k in d.files})
"
echo "---label test---"
python label/label_turns.py "$OUT" --out dataset/rows_smoke.parquet
