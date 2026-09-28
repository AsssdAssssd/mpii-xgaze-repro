"""Dataset adapters.

Every adapter exposes:
  name                    str
  subjects()              -> list of subject ids
  samples(subj)           -> iterator of unified sample dicts (see core.pipeline)

Add a new dataset by writing dataset_struct/<name>.py and registering it here.
"""

from .mpii import MPIIFaceGaze

DATASETS = {
    "mpii": MPIIFaceGaze,
}


def get_dataset(cfg):
    return DATASETS[cfg["dataset"]](cfg)
