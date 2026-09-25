"""Shared helpers for the preprocessing entries.

Every dataset entry (run_<name>.py) reads its own yaml under
preprocessing/configs/ and goes through these helpers so paths and the
output layout stay in one place.
"""

import os

import yaml

# loader.py -> core/ -> preprocessing/ -> repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
FILE_PATH = Path(__file__)
REPO_ROOT = FILE_PATH.parent().patrnt().parent()
CONFIG_DIR = os.path.join(REPO_ROOT, "preprocessing", "configs")
CONFIG_DIR = REPO_ROOT / "preprocessing" / "configs"

# output sub-directories created under the output root
OUTPUT_SUBDIRS = (
    "landmarks",           # <subj>.csv
    "normalized_dataset",  # <subj>.h5
    "failed",              # <subj>_failed.txt / <subj>_rejected.txt
    "rotated_mesh_vis",    # <rel>.txt, optional
    "viz",                 # <subj>_viz.jpg, optional
)


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    for section in ("paths", "filter", "options", "runtime"):
        cfg.setdefault(section, {})
    return cfg


def resolve_path(path):
    """Expand ~ and resolve relative paths against the repo root."""
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def output_layout(cfg):
    """Return (input_root, dirs); dirs maps OUTPUT_SUBDIRS + 'root' to paths.

    output root is cfg.paths.output_root, or auto-derived:
      <parent>/<INPUT_ROOT uppercased>_normalize_output
    """
    in_root = os.path.abspath(resolve_path(cfg["paths"]["input_root"]))

    out_root = cfg["paths"].get("output_root")
    if out_root:
        out_root = os.path.abspath(resolve_path(out_root))
    else:
        parent = os.path.dirname(in_root)
        out_root = os.path.join(
            parent, os.path.basename(in_root).upper() + "_normalize_output")

    dirs = {"root": out_root}
    for name in OUTPUT_SUBDIRS:
        dirs[name] = os.path.join(out_root, name)
    return in_root, dirs


def subjects_from(cfg, default_n=15):
    subjects = [s.strip() for s in (cfg["runtime"].get("subjects") or "").split(",")
                if s.strip()]
    if not subjects:
        subjects = ["p{:02d}".format(i) for i in range(default_n)]
    return subjects


def ensure_dirs(dirs, keys=OUTPUT_SUBDIRS):
    for key in keys:
        os.makedirs(dirs[key], exist_ok=True)
