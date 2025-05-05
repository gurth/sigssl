import torch.utils.data as data
import torch
import numpy as np
import torch
import json
import os
import cv2
import math

from utils.radio import read_mod_lab
# from utils.radio import wavelet_decomposition

class SigDetDataset(data.Dataset):
    def __getitem__(self, index):
        index = self.idx[index]

        bin_path, lab_path = self.get_path(index)

        # Read IQ sample data and label from file
        data, anns = read_mod_lab(bin_path, lab_path)

        # Normalize data and label
        inp = (data - self.mean) / self.std

        # To tensor
        inp = torch.from_numpy(inp).float()
        shape = inp.shape

        ret = {}
        num_objs = min(len(anns), self.max_objs)

        # Initialize "boxes" and "labels" with appropriate dimensions
        ret["boxes"] = torch.zeros((self.max_objs, 2), dtype=torch.float32)
        ret["labels"] = torch.zeros((self.max_objs,), dtype=torch.long)

        for i in range(num_objs):
            ann = anns[i]
            t_bbox = [(ann["bbox"][1] + ann["bbox"][3]) / 2, -ann["bbox"][1] + ann["bbox"][3]]
            t_bbox = [t_bbox[0] / shape[0], t_bbox[1] / shape[0]]

            ret["boxes"][i] = torch.tensor(t_bbox)
            cls_id = 1 if self.opt.no_cls else self.class_name.index(ann['category'])
            ret["labels"][i] = cls_id

        # Trim unused slots if less than max_objs
        if num_objs < self.max_objs:
            ret["boxes"] = ret["boxes"][:num_objs]
            ret["labels"] = ret["labels"][:num_objs]

        # Ensure no empty tensors
        if num_objs == 0:
            ret["boxes"] = torch.empty((0, 2), dtype=torch.float32)
            ret["labels"] = torch.empty((0,), dtype=torch.long)

        ret["id"] = torch.tensor([index], dtype=torch.long)
        ret['size'] = torch.tensor([shape[0]], dtype=torch.long)

        return inp, ret