import torch
import math
import sys
import time
import os
from progress.bar import Bar

from .base_trainer import BaseTrainer
from datasets.utils.data_prefetcher import data_prefetcher
from datasets.utils.rod_eval import RODEvaluator

from utils.utils import AverageMeter, SmoothedValue, MetricLogger
from utils.misc import get_total_grad_norm
from utils.box_ops import box_cxw_to_xx

class SigdetTrainer(BaseTrainer):
    def __init__(self, opt, model, criterion, postprocessors, base_ds, swav_model, optimizer=None, max_norm=0.0):
        super(SigdetTrainer, self).__init__(opt, model, criterion, optimizer)
        self.device = opt.device
        self.postprocessors = postprocessors
        self.base_ds = base_ds

        self.save_result = opt.save_result

        self.clip_max_norm = opt.clip_max_norm
        self.swav_model = swav_model
        self.max_norm = max_norm

        if opt.task == 'sigdet' or opt.task == 'selfdet':

            if opt.model == 'detr':
                self.detail_loss = {'loss_ce': "ce", 'loss_bbox': "bbox", 'loss_giou': "giou"}
            elif opt.model == 'deformable_detr':
                self.detail_loss = {'loss_ce': "ce", 'loss_bbox': "bbox", 'loss_giou': "giou"}

        if opt.object_embedding_loss:
            self.detail_loss['loss_object_embedding'] = "obj_emb"

    def run_epoch(self, phase, epoch, data_loader):
        if phase == 'train':
            self.model.train()
            self.criterion.train()
        elif phase == 'val':
            self.model.eval()
            self.criterion.eval()

        avg_loss = AverageMeter()
        avg_detail_loss = {k: AverageMeter() for k in self.detail_loss}

        batch_time = AverageMeter()
        num_iters = len(data_loader)

        bar = Bar('{}'.format(self.opt.task), max=num_iters)
        end = time.time()

        # metric_logger = MetricLogger(delimiter="  ")
        # metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
        # metric_logger.add_meter('grad_norm', SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 10

        iou_types = tuple(k for k in ('bbox', #'segm',
                                      ) if k in self.postprocessors.keys())
        rod_evaluator = RODEvaluator(iou_types)

        prefetcher = data_prefetcher(data_loader, self.device , prefetch=True)
        samples, targets = prefetcher.next()

        #for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):

        for iter_id in range(len(data_loader)):
            outputs = self.model(samples)

            if self.swav_model is not None:
                with torch.no_grad():
                    for elem in targets:
                        patches = elem['patches']
                        patches = patches.permute(0, 2, 1)
                        patches = self.swav_model(patches)
                        elem['patches'] = patches
                        del patches

            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # # reduce losses over all GPUs for logging purposes
            # loss_dict_reduced = misc.reduce_dict(loss_dict)
            # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
            #                               for k, v in loss_dict_reduced.items()}
            # loss_dict_reduced_scaled = {k: v * weight_dict[k]
            #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
            # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            #
            # loss_value = losses_reduced_scaled.item()

            loss_value = losses.item()

            if phase == 'train':

                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    # print(loss_dict_reduced)
                    sys.exit(1)

                self.optimizer.zero_grad()
                losses.backward()
                if self.max_norm > 0:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                else:
                    grad_total_norm = get_total_grad_norm(self.model.parameters(), self.max_norm)
                self.optimizer.step()

            elif phase == 'val':
                orig_target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                outputs_results = self.postprocessors['bbox'](outputs, orig_target_sizes)

                if self.save_result:
                    det_result_dir = os.path.join(self.output_dir, "det_result")
                    os.makedirs(det_result_dir, exist_ok=True)
                    for i, target in enumerate(targets):
                        id = target['id'].item()
                        pred_logits = outputs['pred_logits'][i]
                        pred_boxes = outputs['pred_boxes'][i]
                        l = target['size']
                        pred_boxes_ = box_cxw_to_xx(pred_boxes) * torch.stack([l, l], dim=-1)
                        torch.save(dict(id=id, target=target, pred_logits=pred_logits, pred_boxes=pred_boxes,
                                        pred_boxes_=pred_boxes_), os.path.join(det_result_dir, str(id) + '.pt'))

                # if 'segm' in self.postprocessors.keys():
                #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                #     results = self.postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)

                outputs_res = {target['id'].item(): output for target, output in zip(targets, outputs_results)}
                targets_res = {}
                for target in targets:
                    target['boxes'] = box_cxw_to_xx(target['boxes']) * torch.stack([target['size'], target['size']], dim=-1)
                    targets_res[target['id'].item()] = target

                rod_evaluator.update(outputs_res, targets_res)

            avg_loss.update(loss_value, samples.tensors.size(1))

            # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            # metric_logger.update(class_error=loss_dict_reduced['class_error'])
            # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            # metric_logger.update(grad_norm=grad_total_norm)

            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter_id, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)
            Bar.suffix = Bar.suffix + '|{} {:.4f} '.format('Loss', avg_loss.avg)

            for k, v in self.detail_loss.items():
                avg_detail_loss[k].update(loss_dict[k].item(), samples.tensors.size(1))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(v, avg_detail_loss[k].avg)

            bar.next()

            samples, targets = prefetcher.next()

            del outputs, losses, loss_dict, weight_dict


        if phase == 'val':
            print("\n")

            # rod_evaluator.synchronize_between_processes()
            rod_evaluator.accumulate()
            rod_evaluator.summarize()

            if 'bbox' in self.postprocessors.keys():
                stats = rod_evaluator.eval['bbox'].stats.tolist()

            print("bbox stats: \n", stats)

        bar.finish()

        ret = {k: v.avg for k, v in avg_detail_loss.items()}
        ret['loss'] = avg_loss.avg
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        if phase == 'val':
            ret['eval_stats'] = stats

        return ret

        # print("Averaged stats:", metric_logger)
        # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
