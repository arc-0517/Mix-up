import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from torch.autograd import Variable
from torch_trainer.utils import mixup_data, mixup_criterion, rand_bbox

class Trainer(object):
    def __init__(self,
                 model: nn.Module):
        super(Trainer, self).__init__()

        self.model = model
        self.compiled = False

    def compile(self,
                ckpt_dir: str,
                loss_function: str = 'ce',
                optimizer: str = 'adam',
                scheduler: str = 'cosine',
                learnig_rate: float = 1e-4,
                epochs: int = 300,
                local_rank: int = 0,
                mixup_type: str = 'mix_up',
                mixup_alpha: float = 1.0,
                mixup_hidden: bool = False):

        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.epochs = epochs
        # self.local_rank = local_rank
        self.device = f"cuda:{local_rank}"

        # to GPU
        self.model.to(self.device)

        # Define Optimizer
        if optimizer == 'adam':
            self.optimizer = optim.Adam(params=self.model.parameters(),
                                        lr=learnig_rate, weight_decay=1e-5)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(params=self.model.parameters(),
                                        lr=learnig_rate, weight_decay=1e-5)
        elif optimizer == 'adamW':
            self.optimizer = optim.AdamW(params=self.model.parameters(),
                                        lr=learnig_rate, weight_decay=1e-5)
        else:
            raise ValueError('Provide a proper optimizer name')

        if scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=0.0001)

        # Define Loss function
        if loss_function == "ce":
            self.loss_function = nn.CrossEntropyLoss()
        elif loss_function == "mse":
            self.loss_function = nn.MSELoss()
        else:
            raise ValueError('Provide a proper loss function name')

        self.compiled = True

        self.mixup_type = mixup_type
        self.mixup_alpha = mixup_alpha
        self.mixup_hidden = mixup_hidden


    def train(self, epoch, data_loader):

        if not self.compiled:
            raise RuntimeError('Training not prepared')

        self._set_learning_phase(True)
        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

        for inputs, targets in train_bar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if self.mixup_type == "mix_up":
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=self.mixup_alpha, use_cuda=self.device)
                inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
                pred = self.model(inputs)
                loss = mixup_criterion(self.loss_function, pred, targets_a, targets_b, lam)
                # loss = self.loss_function(pred, targets)

            elif self.mixup_type == "cut_mix":
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                rand_idx = torch.randperm(inputs.size()[0]).to(self.device)
                bbx1, bbx2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2] = inputs[rand_idx, :, bbx1:bbx2]
                targets_a = targets
                targets_b = targets[rand_idx]
                pred = self.model(inputs)
                loss = mixup_criterion(self.loss_function, pred, targets_a, targets_b, lam)

            elif self.mixup_type == 'manifold_mixup':
                pred, targets_a, targets_b, lam = self.model(inputs, target=targets, mixup_hidden=True, use_cuda=self.device)
                loss = mixup_criterion(self.loss_function, pred, targets_a, targets_b, lam).mean()

            elif self.mixup_type == 'no_mixup':
                pred = self.model(inputs)
                loss = self.loss_function(pred, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_num += data_loader.batch_size
            total_loss += loss.item() * data_loader.batch_size
            train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch,
                                                                                              self.epochs,
                                                                                              self.get_lr(self.optimizer),
                                                                                              total_loss / total_num))
        return total_loss / total_num

    @torch.no_grad()
    def valid(self, epoch, data_loader):
        pred_list, label_list = [], []
        self._set_learning_phase(True)
        total_loss, total_num, valid_bar = 0.0, 0, tqdm(data_loader)

        for inputs, targets in valid_bar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pred = self.model(inputs)

            _, predicted = torch.max(pred.data, 1)
            pred_list += predicted.tolist()
            label_list += targets.tolist()

            loss = self.loss_function(pred, targets)
            valid_acc = accuracy_score(pred_list, label_list)

            total_num += data_loader.batch_size
            total_loss += loss.item() * data_loader.batch_size
            valid_bar.set_description('Valid Epoch: [{}/{}], Loss: {:.4f}, Acc: {:.3f}'.format(epoch,
                                                                                               self.epochs,
                                                                                               total_loss / total_num,
                                                                                               valid_acc))
        return total_loss / total_num, valid_acc


    @torch.no_grad()
    def test(self, epoch, data_loader):
        pred_list, label_list = [], []
        self._set_learning_phase(False)
        test_bar = tqdm(data_loader)

        for inputs, targets in test_bar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pred = self.model(inputs)

            _, predicted = torch.max(pred.data, 1)
            pred_list += predicted.tolist()
            label_list += targets.tolist()

            test_acc = accuracy_score(pred_list, label_list)
            test_bar.set_description('Test Epoch: [{}/{}], '
                                     'ACC: {:.3f}'.format(epoch,
                                                          self.epochs,
                                                          test_acc))
        return test_acc

    def save(self, epoch, results):
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(self.ckpt_dir + '/log.csv', index_label='epoch')
        torch.save({'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()},
                    self.ckpt_dir + '/model_last.pth')


    def _set_learning_phase(self, train: bool = False):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']
