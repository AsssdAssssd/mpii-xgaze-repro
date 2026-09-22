import argparse
import os
import shutil

import numpy as np
import torch
import yaml

from trainer import Trainer
from data_loader import get_train_loader, get_test_loader


def load_config(config_path, mode):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    project_name = config["experiment"]["name"]
    project_root = config["experiment"]["root"].format(name=project_name)
    os.makedirs(os.path.join(f"{project_root}"),exist_ok=True)
    mode_dir = os.path.join(project_root, mode)

    if mode=="train":
        os.makedirs(os.path.join(mode_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(mode_dir, "weights"), exist_ok=True)
    else:
        os.makedirs(os.path.join(mode_dir, "output"), exist_ok=True)

    shutil.copy(config_path, os.path.join(project_root, "config.yaml"))

    return config


def run(config,is_train):
    kwargs = {}
    if config["experiment"]["use_gpu"]:
        # ensure reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        np.random.seed(0)
        kwargs = {'num_workers': config["experiment"]["num_workers"]}

    data_dir = config["experiment"]["data_dir"]
    batch_size = config["experiment"]["batch_size"]

    # instantiate data loaders
    if is_train:
        data_loader = get_train_loader(
            data_dir, batch_size, is_shuffle=True, **kwargs)
    else:
        data_loader = get_test_loader(
            data_dir, batch_size, is_shuffle=False, **kwargs)
        is_eval=config["test"]["is_eval"]
        eval_path=config["test"]["eval_label"]

    # instantiate trainer
    trainer = Trainer(config, data_loader, is_train)

    # either train
    if is_train:
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()
        if is_eval:
            trainer.evaluation(eval_path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="./configs/config.yaml", type=str)
    p.add_argument("--mode", choices=["train", "test"], default="train")
    args = p.parse_args()

    config = load_config(args.config, args.mode)
    run(config,args.mode=="train")
