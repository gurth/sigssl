import torch
import time
from progress.bar import Bar
from utils.utils import AverageMeter

from datasets.utils.rod_eval import RODEvaluator

class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'], batch['input_lengths'])
        loss, loss_stats = self.loss(outputs, batch)
        return outputs, loss, loss_stats


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)

    def set_device(self, device):
        self.model_with_loss = self.model_with_loss.to(device)

        # for state in self.optimizer.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.to(device=device, non_blocking=True)

    def run_epoch(self, phase, epoch, data_loader):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()
        opt = self.opt

        results = {}
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        bar = Bar('{}'.format(opt.task), max=num_iters)
        end = time.time()

        rod_evaluator = RODEvaluator(opt, ["bbox"])

        for iter_id, inp in enumerate(data_loader):
            batch, targets = inp
            if iter_id >= num_iters:
                break
            for k in batch:
                batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if phase == 'val':
                outputs_results = self.run_val(output, model_with_loss.model)

                # import matplotlib.pyplot as plt
                # import numpy as np
                # def plot_proposals(iq_data, proposals):
                #     plt.figure(figsize=(10, 6))
                #
                #     # Plot the I/Q data
                #     plt.plot(np.real(iq_data), label='I Component')
                #     plt.plot(np.imag(iq_data), label='Q Component')
                #
                #     # Plot the proposals
                #     for proposal in proposals:
                #         start, end = proposal
                #         plt.axvspan(start, end, color='red', alpha=0.3)
                #
                #     plt.xlabel('Sample Index')
                #     plt.ylabel('Amplitude')
                #     plt.title('I/Q Data with Proposals')
                #     plt.legend()
                #     plt.show()
                #     plt.savefig("tmp.png")

                # plot_proposals(targets[0]['iq'], targets[0]['boxes'])

                outputs_res = {target['id'].item(): output for target, output in zip(targets, outputs_results)}



                targets_res = {}
                for target in targets:
                    targets_res[target['id'].item()] = target


                rod_evaluator.update(outputs_res, targets_res)

                # if iter_id == 1:
                #     rod_evaluator.synchronize_between_processes()
                #     rod_evaluator.accumulate()
                #     rod_evaluator.summarize()
                #
                # if iter_id == 10:
                #     rod_evaluator.synchronize_between_processes()
                #     rod_evaluator.accumulate()
                #     rod_evaluator.summarize()

            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)


            bar.next()

            del output, loss, loss_stats

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.

        stats = None
        if phase == 'val':
            print("\n")

            rod_evaluator.synchronize_between_processes()
            rod_evaluator.accumulate()
            rod_evaluator.summarize()

            stats = rod_evaluator.eval['bbox'].stats.tolist()

            print("bbox stats: \n", stats)

        bar.finish()

        return ret, results, stats

    def run_val(self, output, model):
        raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)

    def _get_losses(self, opt):
        raise NotImplementedError