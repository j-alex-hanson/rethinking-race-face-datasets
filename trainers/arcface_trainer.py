import numpy as np
import time
import torch
from torch.nn import DataParallel, Linear
from torch.optim.lr_scheduler import StepLR

from models import AddMarginProduct, ArcMarginProduct, SphereProduct
from .trainer import Trainer

class ArcfaceTrainer(Trainer):
    
    def __init__(self, opt, *args, **kwargs):
        super(ArcfaceTrainer, self).__init__(opt, *args, **kwargs)

        if opt['metric'] == 'add_margin':
            self.metric_fc = AddMarginProduct(512, opt['num_classes'], s=30, m=0.35)
        elif opt['metric'] == 'arc_margin':
            self.metric_fc = ArcMarginProduct(512, opt['num_classes'], s=30, m=0.5, easy_margin=opt['easy_margin'])
        elif opt['metric'] == 'sphere':
            self.metric_fc = SphereProduct(512, opt['num_classes'], m=4)
        else:
            self.metric_fc = Linear(512, opt['num_classes'])

        self.metric_fc.cuda()
        self.metric_fc = DataParallel(self.metric_fc)

        if opt['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD([{'params':self.model.parameters()}, {'params': self.metric_fc.parameters()}],
                                        lr=opt['lr'], weight_decay=opt['weight_decay'])
        else:
            self.optimizer = torch.optim.Adam([{'params': self.model.parameters()}, {'params': self.metric_fc.parameters()}],
                                         lr=opt['lr'], weight_decay=opt['weight_decay'])
        self.scheduler = StepLR(self.optimizer, step_size=opt['lr_step'], gamma=0.1)

        self.load_model()


    def train_iteration(self, epoch):
        self.model.train()

        start = time.time()
        for ii, batch in enumerate(self.train_loader):
            images, label = batch
            images = images.cuda()
            label = label.cuda().long()
            feature = self.model(images)
            output = self.metric_fc(feature, label)

            loss = self.criterion(output, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            iters = epoch * len(self.train_loader) + ii

            if iters % self.opt['print_freq'] == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = self.opt['print_freq'] / (time.time() - start)
                minutes_remaining = (len(self.train_loader) * self.max_epoch - iters) / speed / 60
                print(f'train epoch {epoch} iter {ii} {speed} iters/s loss {loss.item()} acc {acc}; estimated {minutes_remaining} minutes remaining')

                start = time.time()

        self.scheduler.step()

        torch.cuda.empty_cache()
