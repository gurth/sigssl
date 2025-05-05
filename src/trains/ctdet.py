# Modified from CenterNet (https://github.com/xingyizhou/CenterNet)
# Copyright (c) 2019 Xingyi Zhou. All rights reserved.

import torch
import numpy as np

from .base_trainer import BaseTrainer
from models.losses import FocalLoss, RegL1Loss
from models.utils import _sigmoid
from models.decode import ctdet_decode

from utils.radio import soft_nms

class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        self.opt = opt
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.crit_wh = self.crit_reg

    def forward(self, output, batch):
        opt = self.opt
        hm_loss, wt_loss, off_loss = 0, 0, 0
        output = {'hm': output.hm, 'wt': output.wt, 'reg': output.reg}

        output['hm'] = _sigmoid(output['hm'])

        hm_loss = self.crit(output['hm'], batch['hm'])
        if opt.wt_weight > 0:
            wt_loss = self.crit_reg(
                output['wt'], batch['reg_mask'],
                batch['ind'], batch['wt'])

        if opt.reg_offset and opt.off_weight > 0:
            off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                      batch['ind'], batch['reg'])

        loss = opt.hm_weight * hm_loss + opt.wt_weight * wt_loss + \
               opt.off_weight * off_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wt_loss': wt_loss, 'off_loss': off_loss}

        return loss, loss_stats
class CtdetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wt_loss', 'off_loss']
        loss = CtdetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        pass

    def save_result(self, output, batch, results):
        pass

    def run_val(self, output, model):
        with torch.no_grad():
            hm = output.hm.sigmoid_()

            wt = output.wt
            reg = output.reg

        meta = {'input_len': 131072, 'output_len': 0}
        if self.opt.nms:
            meta['nms_setting'] = self.opt.nms_setting

        meta['output_len'] = output.hm.shape[2]

        dets = ctdet_decode(hm, wt, reg=reg, K=self.opt.K)
        dets = self.post_process(dets, meta)

        # new_dets = []
        # for det in dets:
        #     det = det[det[:, 2] > self.opt.vis_thresh]
        #     new_dets.append(det)
        # dets = new_dets

        outputs_results = []
        for det in dets:
            outputs_results.append({
                "boxes": det[:, :2],
                "scores": det[:, 2],
                "labels": det[:, 3],
            })
        return outputs_results

    def post_process(self, dets, meta):
        down_ratio = meta['input_len'] / meta['output_len']

        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])

        # Restore the original box size
        dets[:, :, :2] *= down_ratio

        # dets[:,:,0] = np.floor(dets[:,:,0]) if dets[:,:,0] > 0 else 0
        dets[:, :, 0] = np.floor(dets[:, :, 0])
        dets[:, :, 0] = np.clip(dets[:, :, 0], 0, meta['input_len'] - 1)

        # dets[:,:,1] = np.ceil(dets[:,:,1]) if dets[:,:,1] > meta['input_len'] - 1 else meta['input_len'] - 1
        dets[:, :, 1] = np.ceil(dets[:, :, 1])
        dets[:, :, 1] = np.clip(dets[:, :, 1], 0, meta['input_len'] - 1)

        if self.opt.nms:
            dets = soft_nms(dets,
                            Nt=meta['nms_setting']['Nt'],
                            sigma=meta['nms_setting']['sigma'],
                            thresh=meta['nms_setting']['thresh'],
                            method=meta['nms_setting']['method'])

        # Resort
        dets = dets[:, dets[0, :, 2].argsort()[::-1], :]

        return dets
