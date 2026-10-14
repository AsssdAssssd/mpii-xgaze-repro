import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import random
from typing import List
from utils.datasetenhance import DatasetEnhancer

trans_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _list_h5(data_dir, sub_folder):
    return sorted(p for p in (Path(data_dir) / sub_folder).glob("*.h5"))


def get_train_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    sub_folder_use = 'train'
    train_set = EVEH5Dataset(dataset_path=data_dir, keys_to_use=_list_h5(data_dir, sub_folder_use),
                            sub_folder=sub_folder_use, transform=trans, is_shuffle=is_shuffle, is_load_label=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    return train_loader

def get_val_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    sub_folder_use = 'test'
    val_set = EVEH5Dataset(dataset_path=data_dir, keys_to_use=_list_h5(data_dir, sub_folder_use),
                          sub_folder=sub_folder_use, transform=trans, is_shuffle=is_shuffle, is_load_label=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

    return val_loader


def get_test_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True,load_label=False):
    sub_folder_use = 'test'   # 该数据集没有单独 test 目录，用 val
    test_set = EVEH5Dataset(dataset_path=data_dir, keys_to_use=_list_h5(data_dir, sub_folder_use),
                           sub_folder=sub_folder_use, transform=trans, is_shuffle=is_shuffle, is_load_label=load_label)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return test_loader


class EVEH5Dataset(Dataset):
    def __init__(self, dataset_path: str, keys_to_use: List[str] = None, sub_folder='', transform=None, is_shuffle=True,
                 index_file=None, is_load_label=True):
        self.path = dataset_path
        self.hdfs = {}
        self.sub_folder = sub_folder
        self.is_load_label = is_load_label

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0

        for subject_idx in range(0, len(self.selected_keys)):
            self.hdfs[subject_idx] = h5py.File(self.selected_keys[subject_idx], 'r', swmr=True)
            assert self.hdfs[subject_idx].swmr_mode
        self.enhance_rate=1.0
        self.enhancer=DatasetEnhancer(self.enhance_rate)

        # Construct mapping from full-data index to key and person-specific index

        self.items = []

        for subject_idx in range(0, len(self.selected_keys)):
            n = self.hdfs[subject_idx]["face_patch"].shape[0]
            self.items += [
                {
                    "subject_idx": subject_idx, 
                    "frame_idx": i, 
                    "is_enhanced": False
                }
            for i in range(n)]

        num_enhanced = int(len(self.hdfs) * self.enhance_rate)

        for i in random.sample(range(n), num_enhanced):
            item = self.items[i]
            self.items.append(
                {
                    "subject_idx": item["subject_idx"], 
                    "frame_idx": item["frame_idx"], 
                    "is_enhanced": True
                }
            )

        if is_shuffle:
            random.shuffle(self.items)  # random the order to stable the training

        self.hdf = None
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

    def __getitem__(self, global_idx):
        item = self.items[global_idx]
        idx = item["frame_idx"]
        self.hdf = h5py.File(self.selected_keys[item["subject_idx"]])
        assert self.hdf.swmr_mode

        # Get face image
        image = self.hdf['face_patch'][idx, :]
        if item["is_enhanced"]:
            image=self.enhancer.image_light_enhance(image)
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB
        image = self.transform(image)

        step = self.hdf['step_index'][idx].item()
        index = self.hdf['frame_index'][idx].item()
        camera = self.hdf['cam_index'][idx].item()
        key=f"{item["subject_idx"]}/{step}/{camera}/{index}"

        # Get labels
        if self.is_load_label:
            gaze_label = self.hdf['face_gaze'][idx, :]
            gaze_label = gaze_label.astype('float')
            self.hdf.close()
            return key,image, gaze_label
        else:
            self.hdf.close()
            return key,image

    def get_labels_keyed(self):
        # eval用：从源h5里读face_gaze，{key: (n,2)}，行序与h5原生顺序一致
        labels = {}
        for key in self.selected_keys:
            hdf = h5py.File(Path(self.path)/self.sub_folder/key, 'r', swmr=True)
            labels[key] = np.asarray(hdf['face_gaze'][:], dtype=np.float64)
            hdf.close()
        return labels




