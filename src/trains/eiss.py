"""
This code implements the training strategy proposed in:
Wang, Z., Wang, J., Liu, Z., & Qiu, Q. (2023). 
"Energy-inspired self-supervised pretraining for vision models." 
arXiv preprint arXiv:2302.01384.
"""

import torch
import torch.nn as nn
import time
from progress.bar import Bar
from utils.utils import AverageMeter


class EnergyInspiredSelfSupervisedPretrain(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss = nn.SmoothL1Loss(beta=1.0)
        self.model = model

    def set_device(self, device):
        self.loss = self.loss.to(device)
        self.model = self.model.to(device)

    def run_epoch(self, phase, epoch, data_loader):
        model = self.model
        criterion = self.loss

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

            corrupted_input = batch["corrupted_input"]

            for _ in range(opt.eiss_num_steps):
                corrupted_input = corrupted_input.detach()
                corrupted_input.requires_grad_(True)

                energy_score = model(corrupted_input, batch["input_lengths"])[0]

                im_grad = torch.autograd.grad(energy_score.sum(), corrupted_input)[0]

                corrupted_input = corrupted_input - opt.eiss_alpha * im_grad

                loss += criterion(corrupted_input, batch["input"])
                #del energy_score, im_grad

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

            avg_loss_stats.update(loss.mean().item(), batch['input'].size(0))
            Bar.suffix = Bar.suffix + '|{} {:.4f} '.format("loss", avg_loss_stats.avg)

            Bar.suffix = Bar.suffix + '|alpha {:.4f} '.format( opt.eiss_alpha.item())

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
