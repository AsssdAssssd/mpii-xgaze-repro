import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from trainer import Trainer
from data_loader import get_train_loader, get_test_loader,get_loo_loader


def load_config(config_path, mode,is_loo):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # train不读
    if mode != "train":
        return config

    project_name = config["experiment"]["name"]
    project_root = config["experiment"]["root"].format(name=project_name)
    config["experiment"]["root"] = project_root
    root= Path(project_root)
    root.mkdir(parents=True,exist_ok=True)

    # loo train 细节run里面建
    if is_loo:
        archived = Path(project_root)/"config.yaml"
        shutil.copy(config_path, archived)
        return config

    # 普通train
    mode_dir =Path(root)/mode
    mode_dir.mkdir(parents=True,exist_ok=True)
    (Path(mode_dir)/ "logs").mkdir(parents=True,exist_ok=True)
    (Path(mode_dir)/ "weights").mkdir(parents=True,exist_ok=True)

    archived = Path(project_root)/"config.yaml"

    prev = {}
    if archived.is_file():
        try:
            with open(archived) as f:
                prev = yaml.safe_load(f) or {}
        except yaml.YAMLError:
            prev = {}

    shutil.copy(config_path, archived)

    # record which datasets this run used; the loaders read <data_dir>/train_test_split.json
    split_path = Path(config["experiment"]["data_dir"])/ "train_test_split.json"
    recorded = {k: prev[k] for k in
                ("train_split", "train_datasets", "test_split", "test_datasets")
                if k in prev}
    if split_path.is_file():
        with open(split_path) as f:
            split = json.load(f)
        recorded["train_split"] = split_path
        recorded["train_datasets"] = split.get("train", [])
        print(f"recorded train_datasets: {len(recorded['train_datasets'])} entries")
    else:
        print("[warn] split file not found, not recorded:", split_path)

    with open(archived, "a") as f:
        f.write("\n# datasets\n")
        for k in ("train_split", "train_datasets", "test_split", "test_datasets"):
            if k not in recorded:
                continue
            if k.endswith("_split"):
                f.write(f"{k}: {recorded[k]}\n")
            else:
                f.write(f"{k}:\n")
                for n in recorded[k]:
                    f.write(f"- {n}\n")

    return config


def run(config,is_train):
    # ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)
    kwargs = {'num_workers': config["experiment"]["num_workers"]}

    data_dir = config["experiment"]["data_dir"]
    batch_size = config["experiment"]["batch_size"]

    if is_train:
        data_loader = get_train_loader(
            data_dir, batch_size, is_shuffle=True, **kwargs)
        trainer = Trainer(config, data_loader, is_train)
        trainer.train()
    # test只认两个参数：本exp的last_ckpt + config["test"]["test_dataset"]，输出到 exp/{name}/test/{test_dataset.name}_test_result
    else:
        test_dir = Path(config["test"]["test_dataset"])
        data_loader = get_test_loader(
            test_dir, batch_size, is_shuffle=False, **kwargs)

        exp_root = config["experiment"]["root"].format(name=config["experiment"]["name"])
        ckpt = str(Path(exp_root)/"train"/"weights"/"last_ckpt.pth.tar")
        out_root = Path(exp_root)/"test"/f"{test_dir.name}_test_result"

        config["experiment"]["root"] = str(out_root)
        config["test"]["pre_trained_model_path"] = ckpt
        (out_root/"test"/"output").mkdir(parents=True, exist_ok=True)

        trainer = Trainer(config, data_loader, False)
        trainer.test()
        trainer.evaluation(True)


def process_loo_eval_results(eval_result, keys,root,name):

    eval_file = Path(root) / f"{name}.txt"
    loo_keys = [k for k in eval_result if k.startswith("fold_")]
    with open(eval_file, "w") as f:
        f.write("Leave-One-Out Evaluation Results:\n")
        for i, (model,(error,error_std)) in enumerate(eval_result.items()):
            key = keys[i] if keys else None
            if key:
                f.write(f"{model} (test={key}): gaze_error: {error:.6f},   gaze_error_std: {error_std:.6f}\n")
            else:
                f.write(f"{model}: gaze_error: {error:.6f},   gaze_error_std: {error_std:.6f}\n")
        f.write("\n")
        if loo_keys:
            mean_error = sum([eval_result[k][0] for k in loo_keys]) / len(loo_keys)
            mean_error_std = sum([eval_result[k][1] for k in loo_keys]) / len(loo_keys)
            f.write(f"Mean of {len(loo_keys)} loo folds: \ngaze_error: {mean_error:.6f}\n gaze_error_std: {mean_error_std:.6f}\n")
    print(f"Leave-One-Out evaluation results saved to {eval_file}")


def run_loo(config,is_train):
    # ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)
    kwargs = {'num_workers': config["experiment"]["num_workers"]}

    data_dir = config["experiment"]["data_dir"]
    batch_size = config["experiment"]["batch_size"]
    ori_root=config["experiment"]["root"].format(name=config["experiment"]["name"])

    if is_train:
        train_data_loaders,test_data_loaders,full_test_loader,keys=get_loo_loader(data_dir, batch_size, **kwargs)
        eval_result={}
        for i in range(len(train_data_loaders)):
            config["experiment"]["root"]=ori_root+"/train"+f"/fold_{i}"
            for sub in ("train/weights", "train/logs","test/output"):
                Path(ori_root+"/train"+f"/fold_{i}", sub).mkdir(parents=True, exist_ok=True)
            trainer = Trainer(config, train_data_loaders[i], True)
            trainer.train()
            config["test"]["pre_trained_model_path"] = str(
                Path(config["experiment"]["root"])/"train"/"weights"/"last_ckpt.pth.tar")
            tester = Trainer(config, test_data_loaders[i], False)
            tester.test()
            error,error_std=tester.evaluation(False)
            eval_result[f"fold_{i}"] = (error, error_std)
        config["experiment"]["root"]=ori_root+"/train"+"/full"
        for sub in ("train/weights", "train/logs","test/output"):
            Path(ori_root+"/full", sub).mkdir(parents=True, exist_ok=True)
        full_trainer = Trainer(config, full_test_loader, True)
        full_trainer.train()
        process_loo_eval_results(eval_result,keys,ori_root,"loo_eval_results")

    # loo test：用外部test_dataset跑fold*+full全部权重，每模型一个子目录，最后汇总
    else:
        test_dir = Path(config["test"]["test_dataset"])
        data_loader = get_test_loader(
            test_dir, batch_size, is_shuffle=False, **kwargs)

        out_root = Path(ori_root)/"test"/f"{test_dir.name}_test_result"
        out_root.mkdir(parents=True, exist_ok=True)
        trained_dir = sorted([i for i in (Path(ori_root)/"train").iterdir()
                              if i.is_dir() and (i.name.startswith("fold_") or i.name=="full")])
        eval_result={}
        for i in trained_dir:
            ckpt = i/"train"/"weights"/"last_ckpt.pth.tar"
            if not ckpt.is_file():
                print("[warn] no ckpt, skip:", ckpt)
                continue
            config["experiment"]["root"] = str(out_root/i.name)
            config["test"]["pre_trained_model_path"] = str(ckpt)
            (out_root/i.name/"test"/"output").mkdir(parents=True, exist_ok=True)

            trainer = Trainer(config, data_loader, False)
            trainer.test()
            error,error_std=trainer.evaluation(True)
            eval_result[i.name] = (error, error_std)
        process_loo_eval_results(eval_result,None,out_root,f"{test_dir.name}_eval_results")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str)
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--loo",action="store_true")
    #test的两个参数：CLI直传/config
    p.add_argument("--test_dataset", default=None, type=str)
    p.add_argument("--pretrained_expname", default=None, type=str)
    args = p.parse_args()

    if args.mode == "test":
        src = load_config(args.config, args.mode,args.loo) if args.config is not None else {}
        expname = args.pretrained_expname or src.get("test", {}).get("pretrained_expname")
        test_dataset = args.test_dataset or src.get("test", {}).get("test_dataset")
        if not expname or not test_dataset:
            p.error("test needs pretrained_expname and test_dataset: pass both, or give a --config with them")

        
        if not Path(f"./exp/{expname}").is_dir():
            p.error(f"pretrained exp not found: ./exp/{expname}")
        # 其余一切以依附exp的config为准
        archived = Path(f"./exp/{expname}")/"config.yaml"
        if not archived.is_file():
            p.error(f"attached exp config not found: {archived}")
        with open(archived) as f:
            config = yaml.safe_load(f)

        exp_root = config["experiment"]["root"].format(name=expname)
        out_root = Path(exp_root)/"test"/f"{Path(test_dataset).name}_test_result"
        out_root.mkdir(parents=True, exist_ok=True)
        shutil.copy(archived, out_root/"config.yaml")
        with open(out_root/"config.yaml") as f:
            config = yaml.safe_load(f)
        config.setdefault("test", {})
        config["test"]["pretrained_expname"]=expname
        config["test"]["test_dataset"]=test_dataset
        with open(out_root/"config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    else:
        if args.config is None:
            p.error("--config is required for train")
        config = load_config(args.config, args.mode,args.loo)

    if(args.loo):
        run_loo(config,args.mode=="train")
    else:
        run(config,args.mode=="train")
