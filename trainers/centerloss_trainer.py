import numpy as np
import time
import torch
from torch.nn import DataParallel, Linear
from torch.optim.lr_scheduler import StepLR

from models import compute_center_loss, get_center_delta
from .trainer import Trainer

class CenterlossTrainer(Trainer):

    def __init__(self, opt, *args, **kwargs):
        super(CenterlossTrainer, self).__init__(opt, *args, **kwargs)

        if opt['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=opt['lr'], weight_decay=opt['weight_decay'])
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=opt['lr'], weight_decay=opt['weight_decay'])

        self.scheduler = StepLR(self.optimizer, step_size=opt['lr_step'], gamma=0.1)
        self.metric_fc = Linear(512, opt['num_classes'])
        self.metric_fc.cuda()
        self.metric_fc = DataParallel(self.metric_fc)
        self.load_model()


    def train_iteration(self, epoch):
        self.model.train()

        start = time.time()
        for ii, batch in enumerate(self.train_loader):
            images, targets = batch
            images = images.cuda()
            targets = targets.cuda().long()
            centers = self.model.module.centers

            features = self.model(images)
            output = self.metric_fc(features)

            cross_entropy_loss = self.criterion(output, targets)
            center_loss = compute_center_loss(features, centers, targets)
            loss = self.opt['lda'] * center_loss + cross_entropy_loss

            center_deltas = get_center_delta(
                features.data, centers, targets, self.opt['alpha'])
            self.model.module.centers = centers - center_deltas

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            iters = epoch * len(self.train_loader) + ii

            if iters % self.opt['print_freq'] == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = targets.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = self.opt['print_freq'] / (time.time() - start)
                minutes_remaining = (len(self.train_loader) * self.max_epoch - iters) / speed / 60
                print(f'train epoch {epoch} iter {ii} {speed} iters/s loss {loss.item()} acc {acc}; estimated {minutes_remaining} minutes remaining')

                start = time.time()

        self.scheduler.step()
