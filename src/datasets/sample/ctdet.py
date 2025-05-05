import torch.utils.data as data
import numpy as np
import torch
import json
import os
import cv2
import math

from utils.radio import read_mod_lab, gaussian_radius, draw_umich_gaussian
from utils.radio import wavelet_decomposition

class CTDetDataset(data.Dataset):
    def __getitem__(self, index):
        index = self.idx[index]

        bin_path, lab_path = self.get_path(index)

        # Read IQ sample data and label from file
        data, anns = read_mod_lab(bin_path, lab_path)
        num_objs = min(len(anns), self.max_objs)

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
            ann = anns[k]
            bbox = np.array(ann['bbox'], dtype=np.float32)
            cls_id = 0 if self.opt.no_cls else (self.class_name.index(ann['category']) - 1)

            t0 = np.floor(bbox[1] / down_ratio)
            t1 = np.ceil(bbox[3] / down_ratio)

            t = (t1 - t0) if self.opt.not_absolute_offset else (bbox[3] - bbox[1]) / down_ratio

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
                    else ((((bbox[1] + bbox[3]) / (2 * down_ratio) - ct_int) + 1 ) /2)
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

        target = {}
        shape = data.shape
        num_objs = min(len(anns), self.max_objs)

        # Initialize "boxes" and "labels" with appropriate dimensions
        target["boxes"] = torch.zeros((self.max_objs, 2), dtype=torch.float32)
        target["labels"] = torch.zeros((self.max_objs,), dtype=torch.long)

        # target["scores"] = torch.ones((self.max_objs,), dtype=torch.float32)

        for i in range(num_objs):
            ann = anns[i]
            # t_bbox = [(ann["bbox"][1] + ann["bbox"][3]) / 2, -ann["bbox"][1] + ann["bbox"][3]]
            # t_bbox = [t_bbox[0] / shape[0], t_bbox[1] / shape[0]]
            t_bbox =[ann["bbox"][1], ann["bbox"][3]]

            target["boxes"][i] = torch.tensor(t_bbox)
            cls_id = 0 if self.opt.no_cls else self.class_name.index(ann['category'] - 1)
            target["labels"][i] = cls_id


        # Trim unused slots if less than max_objs
        if num_objs < self.max_objs:
            target["boxes"] = target["boxes"][:num_objs]
            target["labels"] = target["labels"][:num_objs]

            # target["scores"] = target["scores"][:num_objs]

        # Ensure no empty tensors
        if num_objs == 0:
            target["boxes"] = torch.empty((0, 2), dtype=torch.float32)
            target["labels"] = torch.empty((0,), dtype=torch.long)

            # target["scores"] = torch.empty((0,), dtype=torch.float32)

        target["id"] = torch.tensor([index], dtype=torch.long)
        target['size'] = torch.tensor([shape[0]], dtype=torch.long)

        target["iq"] = data

        return ret, target
