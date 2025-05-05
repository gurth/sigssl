from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

import torch.utils.data as data


class ROD(data.Dataset):
    # mean = [-1.4995751071756204e-06, 1.3660641390164907e-05]
    mean = [0, 0]
    std = [0.7071117222130318, 0.707102749139276]
    nums_all = 12500 * 4

    num_classes = 10

    freq_max = 6e3
    freq_min = -6e3

    class_name = [
        '__background__', 'BPSK', 'QPSK', 'OQPSK', 'pi/4-DQPSK', '8PSK',
        '16QAM', '64QAM', 'GMSK', '4GFSK', 'OFDM',
    ]

    signal_power = [0.1, 0.2, 0.5, 1, 2, 4, 8, 10, 100]

    num_ext = []

    def __init__(self, opt, split):
        self.opt = opt
        if opt.ext:
            self.num_ext = [1000 * 4]
            self.power_ext = [[0.0625,0.03125,0.015625,0.01]]

        if opt.no_cls:
            self.num_classes = 1
            self.class_name = ['__background__', 'signal']

        self.split = split
        self.data_dir = opt.rod_path

        self.finetune_ratio = opt.finetune_ratio

        self.max_objs = 64

        self.nums_idx = [self.nums_all]

        if len(self.num_ext) > 0:
            for item in self.power_ext:
                self.signal_power = sorted(set(self.signal_power) | set(item))
            for i in range(len(self.num_ext)):
                self.nums_all += self.num_ext[i]
                self.nums_idx.append(self.nums_idx[i] + self.num_ext[i])

        nums_val = round(self.nums_all * 0.2)
        nums_train = self.nums_all - round(self.nums_all * 0.2)

        idx_all = np.arange(self.nums_all)

        np.random.seed(opt.seed)

        idx_train, idx_val = self.split_data(idx_all, nums_val)

        if self.opt.finetune:
            nums_train = len(idx_train)
            nums_val = len(idx_val)

        if self.split == 'train':
            self.num_samples = nums_train
            self.idx = idx_train
        elif self.split == 'val':
            self.num_samples = nums_val
            self.idx = idx_val

        # Selfdet parameters
        self.catIds= [i for i in range(self.num_classes + 1)]

    def split_data(self, idx_all, nums_val):
        idx_val = np.random.choice(idx_all, nums_val, replace=False)
        idx_train = np.setdiff1d(idx_all, idx_val)

        if self.opt.finetune:
            assert 0 < self.finetune_ratio < 1
            idx_train = idx_val[:int(len(idx_val) * self.finetune_ratio)]
            idx_val = idx_val[int(len(idx_val) * self.finetune_ratio):]


        return idx_train, idx_val

    def get_path(self, index):
        bin_path = ""
        lab_path = ""

        if index < self.nums_idx[0]:
            bin_path = os.path.join(self.data_dir, "mod_bin" + os.path.sep + "{:05d}.npy".format(index + 1))
            lab_path = os.path.join(self.data_dir, "mod_lab" + os.path.sep + "{:05d}.npy".format(index + 1))
        else:
            for i, boundary in enumerate(self.nums_idx):
                if index < boundary:
                    bin_path = os.path.join(self.data_dir,
                                            "ext" + os.path.sep + "{}".format(i) + os.path.sep + "mod_bin" + os.path.sep +
                                            "{:05d}.npy".format(index + 1 - self.nums_idx[i - 1]))
                    lab_path = os.path.join(self.data_dir,
                                            "ext" + os.path.sep + "{}".format(i) + os.path.sep + "mod_lab" + os.path.sep +
                                            "{:05d}.npy".format(index + 1 - self.nums_idx[i - 1]))
                    break

        return bin_path, lab_path

    def __len__(self):
        return self.num_samples
