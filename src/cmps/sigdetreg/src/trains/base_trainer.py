import torch
import time
from progress.bar import Bar



class BaseTrainer(object):
    def __init__(self, opt, model, criterion, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.model = model
        self.criterion = criterion

        self.output_dir = opt.output_dir

    def set_device(self, device):
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)

    def run_epoch(self, phase, epoch, data_loader):
        raise NotImplementedError
