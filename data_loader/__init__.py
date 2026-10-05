from .xgaze_data_loader import (
    GazeDataset,
    get_train_loader as _xgaze_train_loader,
    get_val_loader as _xgaze_val_loader,
    get_test_loader as _xgaze_test_loader,
)
from .eve_data_loader import (
    EveDataset,
    get_eve_train_loader as _eve_train_loader,
    get_eve_val_loader as _eve_val_loader,
    get_eve_test_loader as _eve_test_loader,
)
from .mpii_data_loader import (
    MPIIDataset,
    get_train_loader as _mpii_train_loader,
    get_val_loader as _mpii_val_loader,
    get_test_loader as _mpii_test_loader,
    get_loo_loader
)

_TRAIN_LOADERS = {
    "xgaze": _xgaze_train_loader,
    "eve": _eve_train_loader,
    "mpii": _mpii_train_loader,
}
_VAL_LOADERS = {
    "xgaze": _xgaze_val_loader,
    "eve": _eve_val_loader,
    "mpii": _mpii_val_loader,
}
_TEST_LOADERS = {
    "xgaze": _xgaze_test_loader,
    "eve": _eve_test_loader,
    "mpii": _mpii_test_loader,
}


def get_train_loader(data_dir, batch_size, num_workers=4, is_shuffle=True,
                     dataset_type="xgaze"):
    return _TRAIN_LOADERS[dataset_type](data_dir, batch_size,
                                        num_workers=num_workers,
                                        is_shuffle=is_shuffle)


def get_val_loader(data_dir, batch_size, num_workers=4, is_shuffle=True,
                   dataset_type="xgaze"):
    return _VAL_LOADERS[dataset_type](data_dir, batch_size,
                                      num_workers=num_workers,
                                      is_shuffle=is_shuffle)


def get_test_loader(data_dir, batch_size, num_workers=4, is_shuffle=True,
                    dataset_type="xgaze", load_label=True):
    return _TEST_LOADERS[dataset_type](data_dir, batch_size,
                                       num_workers=num_workers,
                                       is_shuffle=is_shuffle, load_label=load_label)
