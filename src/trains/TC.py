# Modified from TS-TCC (https://github.com/emadeldeen24/TS-TCC)
# Copyright (c) 2022 Emadeldeen Eldele. 

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from progress.bar import Bar
from utils.utils import AverageMeter
from models.networks.TC.TC import TC
from models.networks.TC.loss import NTXentLoss

class TCPretrain(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.lambda1 = 1
        self.lambda2 = 0.7
        self.loss= NTXentLoss(opt.device, opt.batch_size, 0.2, True)
        self.model = model

        opt.final_out_channels = 144
        from argparse import Namespace
        opt.TC = Namespace()
        opt.TC.timesteps = 10
        opt.TC.hidden_dim = 100

        temporal_contr_model = TC(opt, opt.device).to(opt.device)
        self.temporal_contr_model = temporal_contr_model

        self.temp_cont_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=opt.lr, betas=(0.9, 0.99), weight_decay=3e-4)

    def set_device(self, device):
        self.loss = self.loss.to(device)
        self.model = self.model.to(device)

    def run_epoch(self, phase, epoch, data_loader):
        model = self.model
        criterion = self.loss
        temporal_contr_model = self.temporal_contr_model

        if phase == 'train':
            model.train()
        else:
            model.eval()
            torch.cuda.empty_cache()
        opt = self.opt

        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = AverageMeter()
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}'.format(opt.task), max=num_iters)
        end = time.time()

        for iter_id, inp in enumerate(data_loader):
            batch, targets = inp
            if iter_id >= num_iters:
                break
            for k in batch:
                batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            loss = 0

            origin_input = batch["input"]
            corrupted_input = batch["corrupted_input"]
            input_lengths = batch["input_lengths"]

            features1 = model(origin_input, input_lengths)
            features2 = model(corrupted_input, input_lengths)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1
            zjs = temp_cont_lstm_feat2
            loss = (temp_cont_loss1 + temp_cont_loss2) * self.lambda1 + criterion(zis, zjs) * self.lambda2

            if phase == 'train':
                self.optimizer.zero_grad()
                self.temp_cont_optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.temp_cont_optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

            avg_loss_stats.update(loss.mean().item(), batch['input'].size(0))
            Bar.suffix = Bar.suffix + '|{} {:.4f} '.format("loss", avg_loss_stats.avg)

            bar.next()

            del loss

        ret = {
            "loss": avg_loss_stats.avg,
            'time': bar.elapsed_td.total_seconds() / 60.
        }
        bar.finish()

        stats = None

        return ret, results, stats

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)
