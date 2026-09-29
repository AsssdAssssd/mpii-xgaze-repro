import argparse
from pathlib import Path

import yaml

from core.pipeline import Pipeline
from dataset_struct import get_dataset

CONFIG_DIR = Path(__file__).resolve().parent/"configs"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset_type", default="mpii",choices=["mpii","eve"])
    p.add_argument("--config", default=None, help="default: configs/<dataset_type>.yaml")
    args = p.parse_args()

    config_path = Path(args.config) if args.config else CONFIG_DIR/f"{args.dataset_type}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    for section in ("paths", "filter", "options", "runtime"):
        cfg.setdefault(section, {})

    cfg["dataset"] = args.dataset_type

    pipeline=Pipeline(cfg, get_dataset(cfg))
    pipeline.run()


if __name__ == "__main__":
    main()
