# Modified from SimCLR (https://github.com/google-research/simclr)

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from progress.bar import Bar
from utils.utils import AverageMeter


class SimCLRPretrain(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss = nn.CrossEntropyLoss().to(opt.device)
        self.model = model

        self.opt.temperature = 0.07

    def info_nce_loss(self, features):
        opt = self.opt

        labels = torch.cat([torch.arange(opt.batch_size) for i in range(opt.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(opt.device)

        features = features.simclr.squeeze(1)
        features = F.normalize(features, dim=1)


        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(opt.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(opt.device)

        logits = logits / opt.temperature
        return logits, labels

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

            inputs = batch["input"]
            corrupted_inputs = batch["corrupted_input"]
            input_lengths = batch["input_lengths"]

            combined_inputs = torch.cat((inputs, corrupted_inputs), dim=0)
            features = model(combined_inputs, input_lengths * 2)
            logits, labels = self.info_nce_loss(features)
            loss = criterion(logits, labels)

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
