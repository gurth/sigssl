import torch.utils.data as data
import numpy as np
import torch
import json
import os
import cv2
import math
import random

from utils.radio import read_mod_lab, gaussian_radius, draw_umich_gaussian
from utils.radio import wavelet_decomposition

from .proposals.proposal_factory import get_proposal

class SelfDetDataset(data.Dataset):
    def init_selfdet(self, cache_dir=None, max_prop=30, strategy='topk'):
        self.strategy = strategy
        self.cache_dir = cache_dir

        self.max_prop = max_prop

        self.dist2 = -np.log(np.arange(1, 301) / 301) / 10
        max_prob = (-np.log(1 / 1001)) ** 4

        self.proposal = get_proposal(strategy, max_prop)

        self.no_cached = []

    def load_from_cache(self, index, radio):
        fn = "{}".format(index) + '.npy'
        fp = os.path.join(self.cache_dir, fn)

        try:
            with open(fp, 'rb') as f:
                boxes = np.load(f)
        except FileNotFoundError:
            boxes = self.proposal(radio)
            with open(fp, 'wb') as f:
                np.save(f, boxes)
        return boxes

    def __getitem__(self, index):
        index = self.idx[index]

        bin_path, lab_path = self.get_path(index)

        # Read IQ sample data and label from file
        data, _ = read_mod_lab(bin_path, lab_path)

        boxes = None
        if self.strategy in self.no_cached:
            boxes = self.proposal(data)
        else:
            boxes = self.load_from_cache(index, data)

        if len(boxes) < 2:
            return self.__getitem__(random.randint(0, self.num_samples - 1))

        num_objs = min(len(boxes), self.max_objs)

        data_len = data.shape[0]

        # Normalize data and label
        data = (data - self.mean) / self.std

        # data = data.reshape(-1)
        # Reshape [131072] to [131072/64, 64]
        # data = data.reshape(-1, self.opt.frame_length)

        # for ann in anns:
        #     # Normalize frequency label
        #     # ann['bbox'][0] = (ann['bbox'][0] - self.freq_min) / (self.freq_max - self.freq_min)
        #     # ann['bbox'][2] = (ann['bbox'][2] - self.freq_min) / (self.freq_max - self.freq_min)
        #     ann['bbox'][1] = np.floor(2 * ann['bbox'][1] / self.opt.frame_length)
        #     ann['bbox'][3] = np.ceil(2 * ann['bbox'][3] / self.opt.frame_length)

        inp = torch.from_numpy(data).float()

        num_classes = self.num_classes

        input_len = int(np.ceil(inp.shape[0] / self.opt.frame_length))
        output_len = (input_len >> 2) - 1
        down_ratio = data_len / output_len

        hm = np.zeros((1 if self.opt.no_cls else num_classes,
                       output_len), dtype=np.float32)
        # f = np.zeros((self.max_objs), dtype=np.float32)
        wt = np.zeros((self.max_objs, 1), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg = np.zeros((self.max_objs, 1), dtype=np.float32)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        draw_gaussian = draw_umich_gaussian

        for k in range(num_objs):
            bbox = boxes[k]
            cls_id = 0

            t0 = np.floor(bbox[0] / down_ratio)
            t1 = np.ceil(bbox[1] / down_ratio)

            t0_origin = bbox[0]
            t1_origin = bbox[1]

            t = (t1 - t0) if self.opt.not_absolute_offset else (t1_origin- t0_origin) / down_ratio

            if t > 0:
                radius = gaussian_radius(t)
                radius = max(0, int(radius))

                ct = (t0 + t1) / 2
                ct_int = int(ct)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wt[k] = 1. * t
                ind[k] = ct_int

                # Use true offset instead of center offset
                reg[k] = (ct - ct_int) if self.opt.not_absolute_offset \
                    else ((((t1_origin + t0_origin) / (2 * down_ratio) - ct_int) + 1 ) /2)
                reg_mask[k] = 1

        if self.opt.wt_decomp:

            if not self.opt.wavelet_setting["cuda"]:
                inp = wavelet_decomposition(inp,
                                            self.opt.wavelet_setting["wavelet"],
                                            self.opt.wavelet_setting["level"]
                                            )
            # input_len = inp.shape[0]


        ret = {'input': inp, 'input_lengths': input_len,
               'wt': wt,
               'reg': reg,
               'hm': hm,
               'ind': ind,
               'reg_mask': reg_mask,
               }

        shape = data.shape
        t = len(data)
        target = {'orig_size': torch.as_tensor([int(t)]), 'size': torch.as_tensor([int(t)]).long()}

        target['boxes'] = torch.from_numpy(np.array(boxes)[:, :2])
        target['boxes'] = target['boxes'] / t

        # target['iscrowd'] = torch.zeros(len(target['boxes']))
        # target["area"] = (target['boxes'][:, 1] - target['boxes'][:, 0]).float()
        target['labels'] = torch.ones(len(target['boxes']), dtype=torch.long)

        target["id"] = torch.tensor([index], dtype=torch.long)

        if len(target['boxes']) < 2:
            return self.__getitem__(random.randint(0, self.num_samples - 1))


        target["iq"] = data

        return ret, target
