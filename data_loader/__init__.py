from .general_data_loader import (
    GazeDataset,
    get_train_loader as _general_train_loader,
    get_val_loader as _general_val_loader,
    get_test_loader as _general_test_loader,
    get_loo_loader,
)
from .eve_data_loader import (
    EveDataset,
    get_eve_train_loader,
    get_eve_val_loader,
    get_eve_test_loader,
)

_TRAIN_LOADERS = {"general": _general_train_loader, "eve": get_eve_train_loader}
_VAL_LOADERS = {"general": _general_val_loader, "eve": get_eve_val_loader}
_TEST_LOADERS = {"general": _general_test_loader, "eve": get_eve_test_loader}


def get_train_loader(data_dir, batch_size, num_workers=4, is_shuffle=True,
                     dataset_type="general"):
    return _TRAIN_LOADERS[dataset_type](data_dir, batch_size,
                                        num_workers=num_workers,
                                        is_shuffle=is_shuffle)


def get_val_loader(data_dir, batch_size, num_workers=4, is_shuffle=True,
                   dataset_type="general"):
    return _VAL_LOADERS[dataset_type](data_dir, batch_size,
                                      num_workers=num_workers,
                                      is_shuffle=is_shuffle)


def get_test_loader(data_dir, batch_size, num_workers=4, is_shuffle=True,
                    dataset_type="general",load_label=True):
    return _TEST_LOADERS[dataset_type](data_dir, batch_size,
                                       num_workers=num_workers,
                                       is_shuffle=is_shuffle,load_label=load_label)
