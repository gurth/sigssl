from torch.utils.data import Dataset
import torch.nn.functional as F
import os
import torch
import numpy as np
import random
import numbers

from .proposals.proposal_factory import get_proposal

from utils.radio import read_mod_lab
from utils.box_ops import box_xx_to_cxw

def get_random_patch_from_radio(radio, min_len=8):
    """
    :param radio: original I/Q data
    :param min_len: min length of the query patch
    :return: query_patch,x,t
    """
    len = radio.shape[0]
    min_x, max_x = min_len, len - min_len
    t = random.randint(min_x, max_x + 1)
    x = np.random.randint(len - t) if len != t else 0
    query_patch = radio[x:x + t]
    return query_patch, x, t

class ResizeIQTransform:
    def __init__(self, target_length=8192):
        self.target_length = target_length

    def __call__(self, sample):
        length, _ = sample.shape
        sample = sample.unsqueeze(0)  # Add batch dimension
        sample = sample.permute(0, 2, 1)  # Change to shape [1, 2, length]
        resized_sample = F.interpolate(sample, size=self.target_length, mode='linear', align_corners=True)
        resized_sample = resized_sample.permute(0, 2, 1).squeeze(0)  # Change back to shape [target_length, 2]
        return resized_sample


def get_query_transforms():
    return ResizeIQTransform(target_length=1024)


class SelfDetDataset(Dataset):
    def init_selfdet(self, cache_dir=None, max_prop=30, strategy='topk'):
        self.strategy = strategy
        self.cache_dir = cache_dir

        self.max_prop = max_prop

        self.dist2 = -np.log(np.arange(1, 301) / 301) / 10
        max_prob = (-np.log(1 / 1001)) ** 4

        self.proposal = get_proposal(strategy, max_prop)
        self.query_transforms = get_query_transforms()

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

        data = (data - self.mean) / self.std

        inp = torch.from_numpy(data).float()

        t = len(inp)

        boxes = None
        if self.strategy in self.no_cached:
            boxes = self.proposal(data)
        else:
            boxes = self.load_from_cache(index, data)

        if len(boxes) < 2:
            return self.__getitem__(random.randint(0, self.num_samples - 1))

        patches = []
        for b in boxes:
            if isinstance(b[0], numbers.Integral) and isinstance(b[1], numbers.Integral):
                patches.append(data[b[0]:b[1]])
            elif isinstance(b[0], numbers.Real) and isinstance(b[1], numbers.Real):
                patches.append(data[int(np.floor(b[0])):int(np.ceil(b[1]))])

        target = {'orig_size': torch.as_tensor([int(t)]), 'size': torch.as_tensor([int(t)]).long()}
        target['patches'] = torch.stack([self.query_transforms(torch.as_tensor(p)) for p in patches], dim=0).float()

        target['boxes'] = torch.from_numpy(np.array(boxes)[:,:2])
        target['boxes'] = target['boxes'] / t
        target['boxes'] = box_xx_to_cxw(target['boxes']).float()

        # target['iscrowd'] = torch.zeros(len(target['boxes']))
        # target["area"] = (target['boxes'][:, 1] - target['boxes'][:, 0]).float()
        target['labels'] = torch.ones(len(target['boxes']), dtype=torch.long)

        target["id"] = torch.tensor([index], dtype=torch.long)

        if len(target['boxes']) < 2:
            return self.__getitem__(random.randint(0, self.num_samples - 1))

        return inp, target
