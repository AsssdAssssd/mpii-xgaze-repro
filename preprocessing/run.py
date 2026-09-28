"""Entry point for preprocessing.

    python preprocessing/run.py --dataset_type mpii
    python preprocessing/run.py --dataset_type mpii --subjects p00,p01 

Each dataset type maps to an adapter in dataset_struct/ which yields samples in
one unified format; the shared core.pipeline then does detect -> PnP -> h5.
"""

import argparse
from pathlib import Path

import yaml

from core import pipeline
from dataset_struct import get_dataset

CONFIG_DIR = Path(__file__).resolve().parent/"configs"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_type", default="mpii")
    p.add_argument("--config", default=None, help="default: configs/<dataset_type>.yaml")
    p.add_argument("--subjects", default=None, help="override runtime.subjects")
    args = p.parse_args()

    config_path = Path(args.config) if args.config else CONFIG_DIR/f"{args.dataset_type}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    for section in ("paths", "filter", "options", "runtime"):
        cfg.setdefault(section, {})

    cfg["dataset"] = args.dataset_type
    if args.subjects is not None:
        cfg["runtime"]["subjects"] = args.subjects


    pipeline.run(cfg, get_dataset(cfg))


if __name__ == "__main__":
    main()
