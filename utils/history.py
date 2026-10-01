"""Append-only writer for the training history (results.jsonl).

Each epoch appends one dict per line, e.g.
    {"epoch": 1, "train_loss": 0.12, "val_ang_mean": 4.21}
Later fields can be added without breaking earlier lines.
"""

import json
from pathlib import Path

import numpy as np


def _json_default(value):
    """Convert numpy scalars/arrays so json.dumps stays happy."""
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class HistorySaver:
    def __init__(self, save_dir, filename="results.jsonl"):
        self.path = Path(save_dir)/filename
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, row):
        """Append one epoch's metrics as a single JSON line."""
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=_json_default) + "\n")
