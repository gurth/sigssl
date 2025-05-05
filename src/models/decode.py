# Modified from CenterNet (https://github.com/xingyizhou/CenterNet)
# Copyright (c) 2019 Xingyi Zhou. All rights reserved.

import numpy
import torch
import torch.nn as nn

from .utils import _gather_feat, _transpose_and_gather_feat

# 1D nms
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool1d(
        heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


# 1D topK
def _topk(scores, K=40):
    # TODO: debug
    batch, cat, length = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % length
    topk_ts = topk_inds.int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()

    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)

    topk_ts = _gather_feat(topk_ts.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ts

def ctdet_decode(heat, wt, reg=None, K=100):
    batch, cat, length = heat.size()

    heat = _nms(heat)
    scores, inds, clses, ts = _topk(heat, K=K)

    # TODO: no absolute

    if reg is not None:
        # TODO: debug
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 1)
        # Denormalize of reg
        reg = reg * 2 - 1
        ts = ts.view(batch, K, 1) + reg[:, :, 0:1]
    else:
        ts = ts.view(batch, K, 1) + 0.5

    wt = _transpose_and_gather_feat(wt, inds)

    wt = wt.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    intervals = torch.cat([ts - wt / 2, ts + wt / 2], dim=2)
    detections = torch.cat([intervals, scores, clses], dim=2)

    return detections

