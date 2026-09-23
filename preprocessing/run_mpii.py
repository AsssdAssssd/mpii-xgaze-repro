"""Entry point for MPIIFaceGaze preprocessing.

    python preprocessing/run_mpii.py
    python preprocessing/run_mpii.py --subjects p00,p01 --limit 20
"""

import argparse
import os

import common
import pipeline
from datasets.mpii import MPIIFaceGaze

DEFAULT_CONFIG = os.path.join(common.CONFIG_DIR, "mpii.yaml")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--subjects", default=None, help="override runtime.subjects")
    p.add_argument("--limit", type=int, default=None, help="override filter.limit")
    args = p.parse_args()

    cfg = common.load_config(args.config)
    if args.subjects is not None:
        cfg["runtime"]["subjects"] = args.subjects
    if args.limit is not None:
        cfg["filter"]["limit"] = args.limit

    pipeline.run(cfg, MPIIFaceGaze(cfg))


if __name__ == "__main__":
    main()
