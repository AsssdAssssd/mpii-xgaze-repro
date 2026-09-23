import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import time
import numpy as np

from utils import AverageMeter, angular_error
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

        self.root= config["experiment"]["root"].format(name=config["experiment"]["name"])

        self.batch_size = config["experiment"]["batch_size"]
        self.use_gpu = config["experiment"]["use_gpu"]
        self.epochs = config["train"]["epochs"]  # the total epoch to train
        # data params
        if is_train:
            self.train_loader = data_loader
            self.num_train = len(self.train_loader.dataset)

            # training params
            self.start_epoch = 0
            self.lr = config["train"]["init_lr"]
            self.lr_patience = config["train"]["lr_patience"]
            self.lr_decay_factor = config["train"]["lr_decay_factor"]
            self.ckpt_dir = os.path.join(f"{self.root}/train/weights")
            self.print_freq = config["train"]["print_freq"]
            self.train_iter = 0
           # configure tensorboard logging
            log_dir = os.path.join(f"{self.root}/train/logs")
            self.writer = SummaryWriter(log_dir=log_dir)

        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
            self.pre_trained_model_path = config["test"]["pre_trained_model_path"].format(root=self.root,epochs=self.epochs)

        if self.use_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

 
        # build model
        self.model = gaze_network(backbone=config["train"]["backbone"])
        if self.use_gpu:
            self.model.cuda()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # initialize optimizer and scheduler only when training
        if is_train:
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr)
            self.scheduler = StepLR(
                self.optimizer, step_size=self.lr_patience, gamma=self.lr_decay_factor)

    def train(self):
        print("\n[*] Train on {} samples".format(self.num_train))
        # train for each epoch
        for epoch in range(self.start_epoch, self.epochs):
            print(
                '\nEpoch: {}/{} - base LR: {:.6f}'.format(
                    epoch + 1, self.epochs, self.lr)
            )

            for param_group in self.optimizer.param_groups:
                print('Learning rate: ', param_group['lr'])

            # train for 1 epoch
            print('Now go to training')
            self.model.train()
            train_acc, loss_gaze = \
                self.train_one_epoch(epoch, self.train_loader)

            # keep only the latest checkpoint (overwrite each epoch)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'scheule_state': self.scheduler.state_dict()
                 }
            )
            self.scheduler.step()  # update learning rate

        self.writer.close()


    def train_one_epoch(self, epoch, data_loader):
        """
        Train the model for 1 epoch of the training set.
        """
        batch_time = AverageMeter()
        errors = AverageMeter()
        losses_gaze = AverageMeter()

        tic = time.time()
        for i, (input_img, target) in enumerate(data_loader):
            input_var = torch.autograd.Variable(input_img.float().cuda())
            target_var = torch.autograd.Variable(target.float().cuda())

            # train gaze net
            pred_gaze= self.model(input_var)

            gaze_error_batch = np.mean(angular_error(pred_gaze.cpu().data.numpy(), target_var.cpu().data.numpy()))
            errors.update(gaze_error_batch.item(), input_var.size()[0])

            loss_gaze = F.l1_loss(pred_gaze, target_var)
            self.optimizer.zero_grad()
            loss_gaze.backward()
            self.optimizer.step()
            losses_gaze.update(loss_gaze.item(), input_var.size()[0])

            if i % self.print_freq == 0:
                self.writer.add_scalar('Loss/gaze', losses_gaze.avg, self.train_iter)

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

                self.writer.add_scalar('Error/train', errors.avg, self.train_iter)

                errors.reset()
                losses_gaze.reset()

            self.train_iter = self.train_iter + 1

        toc = time.time()
        batch_time.update(toc-tic)

        print('running time is ', batch_time.avg)
        return errors.avg, losses_gaze.avg

    def test(self):
        """
        Test the pre-treained model on the whole test set. Note there is no label released to public, you can
        only save the predicted results. You then need to submit the test resutls to our evaluation website to
        get the final gaze estimation error.
        """
        print('We are now doing the final test')
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
        result_path = f'{self.root}/test/output/test_results.txt'
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

    def evaluation(self, eval_path):
        output_dir = os.path.join(f"{self.root}/test/output")

        print('now we begin')

        print('loading truth_file')
        truth = self._load_keyed(eval_path)

        print('loading submission file')
        submission = self._load_keyed(os.path.join(output_dir, "test_results.txt"))

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
        output_filename = os.path.join(output_dir, 'eva_scores.txt')
        with open(output_filename, 'w') as output_file:
            output_file.write("gaze_error: %0.4f\n" % error)
            output_file.write("gaze_error_std: %0.4f\n" % error_std)
        print('gaze_error: ', error)
        print('gaze_error_std: ', error_std)

    def save_checkpoint(self, state):
        """
        Save the latest copy of the model (overwrites the previous epoch).
        """
        filename = 'last_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        print('save file to: ', ckpt_path)

    def load_checkpoint(self, input_file_path='./ckpt/ckpt.pth.tar', is_strict=True):
        """
        Load the copy of a model.
        """
        print('load the pre-trained model: ', input_file_path)
        map_location = 'cuda' if self.use_gpu else 'cpu'
        ckpt = torch.load(input_file_path, map_location=map_location, weights_only=False)

        # load variables from checkpoint
        self.model.load_state_dict(ckpt['model_state'], strict=is_strict)

        print(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                input_file_path, ckpt['epoch'])
        )
