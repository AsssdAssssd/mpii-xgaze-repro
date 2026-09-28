from .mpii import MPIIFaceGaze
from .eve import EVE
DATASETS= {
    "mpii": MPIIFaceGaze,
    "eve":EVE,
}


def get_dataset(cfg):
    return DATASETS[cfg["dataset"]](cfg)
