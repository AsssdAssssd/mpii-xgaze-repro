import torch
import torch.nn.functional as F
import logging
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import time
import numpy as np

from utils import AverageMeter, angular_error, HistorySaver
from model import gaze_network

class Trainer(object):
    def __init__(self, config, data_loader, is_train):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: dict loaded from the yaml config file.
        - data_loader: data iterator
        - is_train: whether to run in training mode
        """

        self.root = Path(config["experiment"]["root"])

        self.batch_size = config["experiment"]["batch_size"]
        self.epochs = config["train"]["epochs"]  # the total epoch to train
        self.logger=logging.getLogger(__name__+Path(config["experiment"]["root"]).name+str(time.time()))
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s -[ %(levelname)s ]- %(message)s')
        self.is_train=is_train
        # data params
        if is_train:
            self.train_loader = data_loader
            self.num_train = len(self.train_loader.dataset)
            # training params
            self.start_epoch = 0
            self.lr = config["train"]["init_lr"]
            self.lr_patience = config["train"]["lr_patience"]
            self.lr_decay_factor = config["train"]["lr_decay_factor"]
            self.ckpt_dir = self.root/"train"/"weights"
            self.print_freq = config["train"]["print_freq"]
            self.train_iter = 0
            self.save_freq = config["train"]["save_freq"]
           # configure tensorboard logging
            log_dir = self.root/"train"/"logs"
            file_handler = logging.FileHandler(log_dir/f"{str(time.time()):.6f}.log")

            self.enable_val = config["train"].get("enable_val", False)
            self.history = HistorySaver(log_dir)

        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
            self.pre_trained_model_path = Path(config["test"]["pre_trained_model_path"].format(root=self.root,epochs=self.epochs))
            log_dir = self.root/"test"/"output"
            file_handler = logging.FileHandler(log_dir/f"{str(time.time()):.6f}.log")
            
            # self.logger.addHandler(logging.StreamHandler())
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        # build model
        self.model = gaze_network(backbone=config["train"]["backbone"])
        self.model.cuda()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # initialize optimizer and scheduler only when training
        if is_train:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr)
            self.scheduler = StepLR(
                self.optimizer, step_size=self.lr_patience, gamma=self.lr_decay_factor)
            self.check_resume()
            
    def check_resume(self):
        """
        Check if a checkpoint exists and resume training from it.
        """
        last_ckpt_path = self.ckpt_dir/"last_ckpt.pth.tar"
        if last_ckpt_path.exists():
            ckpt = torch.load(last_ckpt_path, weights_only=False)
            self.model.load_state_dict(ckpt['model_state'], strict=True)
            self.optimizer.load_state_dict(ckpt['optim_state'])
            self.scheduler.load_state_dict(ckpt['scheule_state'])
            self.start_epoch = ckpt['epoch'] + 1
            print(
                "[*] Resumed from {} @ epoch {}".format(
                    last_ckpt_path, ckpt['epoch'])
            )
            self.logger.info("[*] Resumed from {} @ epoch {}".format(last_ckpt_path, ckpt['epoch']))
            self.scheduler.step()  # update learning rate

    def train(self):
        print("\n[*] Train on {} samples".format(self.num_train))
        train_start = time.time()
        # train for each epoch
        for epoch in range(self.start_epoch, self.epochs):
            epoch_tic = time.time()
            epoch_lr = self.optimizer.param_groups[0]["lr"]
            torch.cuda.reset_peak_memory_stats()
            print(
                '\nEpoch: {}/{} - base LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.lr)
            )
            self.logger.info('Epoch: {}/{} - base LR: {:.6f}'.format(epoch + 1, self.epochs, self.lr))

            for param_group in self.optimizer.param_groups:
                print('Learning rate: ', param_group['lr'])
                self.logger.info('Learning rate: {:.6f}'.format(param_group['lr']))

            # train for 1 epoch
            print('Now go to training')
            self.model.train()
            train_err, train_loss = \
                self.train_one_epoch(epoch, self.train_loader)
            train_time = time.time() - epoch_tic

            self.save_checkpoint(
                {'epoch': epoch ,
                    'model_state': self.model.state_dict(),
                    'optim_state': self.optimizer.state_dict(),
                    'scheule_state': self.scheduler.state_dict()
                    }
            )
            self.scheduler.step()  # update learning rate

            val_time = 0.0
            val_metrics = {}
            if self.enable_val == 0:
                val_tic = time.time()
                val_metrics = self.validate()
                val_time = time.time() - val_tic

            self.history.write({
                "epoch": epoch + 1,
                "step": self.train_iter,
                "lr": epoch_lr,
                "elapsed": time.time() - train_start,
                "train_time": train_time,
                "val_time": val_time,
                "gpu_mem_mb": torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                "train_loss": train_loss,
                "train_err_ang_mean": train_err,
                **val_metrics,
            })


    def train_one_epoch(self, epoch, data_loader):
        """
        Train the model for 1 epoch of the training set.
        """
        batch_time = AverageMeter()
        errors = AverageMeter()
        losses_gaze = AverageMeter()
        # epoch-level meters, never reset, used for the history file
        epoch_errors = AverageMeter()
        epoch_losses = AverageMeter()

        tic = time.time()
        for i, (input_img, target) in enumerate(data_loader):
            input_var = torch.autograd.Variable(input_img.float().cuda())
            target_var = torch.autograd.Variable(target.float().cuda())

            # train gaze net
            pred_gaze= self.model(input_var)

            gaze_error_batch = np.mean(angular_error(pred_gaze.cpu().data.numpy(), target_var.cpu().data.numpy()))
            errors.update(gaze_error_batch.item(), input_var.size()[0])
            epoch_errors.update(gaze_error_batch.item(), input_var.size()[0])

            loss_gaze = F.l1_loss(pred_gaze, target_var)
            self.optimizer.zero_grad()
            loss_gaze.backward()
            self.optimizer.step()
            losses_gaze.update(loss_gaze.item(), input_var.size()[0])
            epoch_losses.update(loss_gaze.item(), input_var.size()[0])

            if i  == 0:
                self.logger.info("angle Error/train: %s  {:.3f} - L1 loss: {:.5f} ".format(errors.avg, losses_gaze.avg))


            # report information
            if i % self.print_freq == 0 and i != 0:
                print('--------------------------------------------------------------------')
                msg = "train error: {:.3f} - loss_gaze: {:.5f}"
                print(msg.format(errors.avg, losses_gaze.avg))

                # measure elapsed time
                print('iteration ', self.train_iter)
                toc = time.time()
                batch_time.update(toc - tic)
                # print('Current batch running time is ', np.round(batch_time.avg / 60.0), ' mins')
                tic = time.time()
                # estimate the finish time
                est_time = (self.epochs - epoch) * (self.num_train / self.batch_size) * (batch_time.avg /self.print_freq)/ 60.0
                print('Estimated training time left: ', np.round(est_time), ' mins')

                self.logger.info("angle Error/train: %s  {:.3f} - L1 loss: {:.5f} ".format(errors.avg, losses_gaze.avg))

                errors.reset()
                losses_gaze.reset()

            self.train_iter = self.train_iter + 1

        toc = time.time()
        batch_time.update(toc-tic)

        print('running time is ', batch_time.avg)
        return epoch_errors.avg, epoch_losses.avg #返回的不再是最后的

    def test(self):
        """
        Test the pre-treained model on the whole test set. Note there is no label released to public, you can
        only save the predicted results. You then need to submit the test resutls to our evaluation website to
        get the final gaze estimation error.
        """
        print('test')
        self.model.eval()
        self.load_checkpoint(is_strict=False, input_file_path=self.pre_trained_model_path)
        pred_gaze_all = np.zeros((self.num_test, 2))
        save_index = 0

        print('Testing on ', self.num_test, ' samples')
        for i, (input_img) in enumerate(self.test_loader):
            input_var = torch.autograd.Variable(input_img.float().cuda())
            pred_gaze = self.model(input_var)
            pred_gaze_all[save_index:save_index+self.batch_size, :] = pred_gaze.cpu().data.numpy()
            save_index += input_var.size(0)

        if save_index != self.num_test:
            print('the test samples save_index ', save_index, ' is not equal to the whole test set ', self.num_test)

        print('Tested on : ', pred_gaze_all.shape[0], ' samples')

        # save predictions grouped by key: a "key:<h5 filename>" header line
        # followed by the "x y" lines belonging to that key
        dataset = self.test_loader.dataset
        result_path = self.root/"test"/"output"/"test_results.txt"
        with open(result_path, 'w') as f:
            cur_key = None
            for i in range(self.num_test):
                key_idx, _ = dataset.idx_to_kv[i]
                filename = dataset.selected_keys[key_idx]
                if filename != cur_key:
                    f.write(f"{filename}:\n")
                    cur_key = filename
                x, y = pred_gaze_all[i]
                f.write(f"{x} {y}\n")
        print('save predictions to ', result_path)
        # self.logger.removeFilter(self.logger.handlers[0])

    @staticmethod
    def _load_keyed(path):
        """Read header + rows, returning {key: np.array([[x, y], ...])}.

        Header may be written as either "<name>:" or "key:<name>".
        """
        data = {}
        cur_key = None
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.endswith(':'):
                    cur_key = line[:-1].strip()
                    data.setdefault(cur_key, [])
                else:
                    data[cur_key].append([float(v) for v in line.split()])
        return {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}

    def evaluation(self, write=True):
        output_dir = self.root/"test"/"output"

        print('now we begin')

        print('loading truth from the source h5')
        truth = self.test_loader.dataset.get_labels_keyed()

        print('loading submission file')
        submission = self._load_keyed(output_dir/"test_results.txt")

        print('now compute the gaze error')
        errors = []
        for key, pred in submission.items():
            if key not in truth:
                print(f'[warn] {key} not found in truth file, skip')
                continue
            gt = truth[key]
            if gt.shape[0] != pred.shape[0]:
                print(f'[warn] {key}: pred {pred.shape[0]} rows vs truth {gt.shape[0]} rows, skip')
                continue
            errors.append(angular_error(pred, gt))
        error_all = np.concatenate(errors)

        error = np.mean(error_all)
        error_std = np.std(error_all)
        
        if write:
            output_filename = output_dir/'eva_scores.txt'
            with open(output_filename, 'w') as output_file:
                output_file.write("gaze_error: %0.4f\n" % error)
                output_file.write("gaze_error_std: %0.4f\n" % error_std)
        print('gaze_error: ', error)
        print('gaze_error_std: ', error_std)
        return error, error_std

    def save_checkpoint(self, state):
        """
        Save the latest copy of the model (overwrites the previous epoch).
        """
        print("start save checkpoint")
        if (state["epoch"] + 1) % self.save_freq == 0:
            #改成隔几个epoch保存一次
            filename = f'epoch_{state["epoch"]}_ckpt.pth.tar'
            ckpt_path = self.ckpt_dir/filename
            torch.save(state, ckpt_path)
            print('save file to: ', ckpt_path)

        #一直覆盖保留最新的，使得单独test可以读最后，中间val也可以读中间产物
        last_ckpt_path = self.ckpt_dir/"last_ckpt.pth.tar"
        torch.save(state, last_ckpt_path)
        print('save checkpoint to: ', last_ckpt_path)
        self.logger.info("save checkpoint to: {}".format(last_ckpt_path))


    def load_checkpoint(self, input_file_path='./ckpt/ckpt.pth.tar', is_strict=True):
        """
        Load the copy of a model.
        """
        input_file_path = Path(input_file_path)
        print('load the pre-trained model: ', input_file_path)
        ckpt = torch.load(input_file_path, weights_only=False)

        # load variables from checkpoint
        self.model.load_state_dict(ckpt['model_state'], strict=is_strict)

        print(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                input_file_path, ckpt['epoch'])
        )

    def set_val_loader(self, val_loader):
        if not self.is_train:
            raise RuntimeError("Cannot set validation loader in test mode.")
        if not self.enable_val:
            raise RuntimeError("Validation loader is not enabled in the configuration.")
        self.val_loader = val_loader

    def validate(self):
        if not hasattr(self, 'val_loader'):
            raise RuntimeError("Validation loader has not been set. Call set_val_loader() first.")

        print("\n[*] Validate on {} samples".format(len(self.val_loader.dataset)))
        self.model.eval()
        losses = AverageMeter()
        errs, preds, gts = [], [], []

        with torch.no_grad():
            for i, (input_img, target) in enumerate(self.val_loader):
                input_var = torch.autograd.Variable(input_img.float().cuda())
                target_var = torch.autograd.Variable(target.float().cuda())

                pred_gaze = self.model(input_var)
                losses.update(F.l1_loss(pred_gaze, target_var).item(), input_var.size()[0])

                pred = pred_gaze.cpu().data.numpy()
                gt = target_var.cpu().data.numpy()
                errs.append(angular_error(pred, gt))
                preds.append(pred)
                gts.append(gt)

        err = np.concatenate(errs)
        pred = np.concatenate(preds)
        gt = np.concatenate(gts)
        rad_to_deg = 180.0 / np.pi

        metrics = {
            "val_loss": losses.avg,
            "val_ang_mean": float(err.mean()),
            "val_pitch_err_mean": float(np.abs(pred[:, 0] - gt[:, 0]).mean() * rad_to_deg),
            "val_yaw_err_mean": float(np.abs(pred[:, 1] - gt[:, 1]).mean() * rad_to_deg),
            "val_ang_median": float(np.median(err)),
            "val_ang_p90": float(np.percentile(err, 90)),
            "val_acc@5": float((err < 5).mean()),
            "val_acc@10": float((err < 10).mean()),
            "val_acc@25": float((err < 25).mean()),
        }
        self.logger.info('Validation: %s', metrics)
        return metrics