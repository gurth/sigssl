import torch.utils.data as data
import numpy as np
import torch
import json
import os
import cv2
import math
import random

from utils.radio import read_mod_lab
from augmenters.augmenter_factory import augmenter_factory
from utils.radio import wavelet_decomposition

class SimCLRDataset(data.Dataset):
    def init_simclr(self):
        self.augmenters = []
        noiser = augmenter_factory["noise"](e=0.5, random_ratio=0)
        flip_iq = augmenter_factory["flip_iq"](random_ratio=0)
        freq_offset = augmenter_factory["freq_offset"](frequency_offset=500, sample_rate=512e3, freq_offset_ratio=0, random_ratio=0)
        random_mask = augmenter_factory["random_mask"](patch_size=32, mask_ratio = 0.1, random_ratio=0)

        self.augmenters.append(noiser)
        self.augmenters.append(flip_iq)
        self.augmenters.append(freq_offset)
        self.augmenters.append(random_mask)

    def __getitem__(self, index):
        index = self.idx[index]

        bin_path, lab_path = self.get_path(index)

        # Read IQ sample data and label from file
        data, _ = read_mod_lab(bin_path, lab_path)

        data_len = data.shape[0]

        # Normalize data and label
        data = (data - self.mean) / self.std

        inp = torch.from_numpy(data).float()
        origin_inp = inp

        input_len = int(np.ceil(inp.shape[0] / self.opt.frame_length))
        output_len = (input_len >> 2) - 1
        down_ratio = data_len / output_len

        origin_corrupted_inp = None
        for aug in self.augmenters:
            origin_corrupted_inp = aug.random_apply(inp)

        origin_corrupted_inp = torch.tensor(origin_corrupted_inp, dtype=torch.float32)

        corrupted_inp = origin_corrupted_inp

        if self.opt.wt_decomp:

            if not self.opt.wavelet_setting["cuda"]:
                inp = wavelet_decomposition(inp,
                                            self.opt.wavelet_setting["wavelet"],
                                            self.opt.wavelet_setting["level"]
                                            )
                corrupted_inp = wavelet_decomposition(origin_corrupted_inp,
                                            self.opt.wavelet_setting["wavelet"],
                                            self.opt.wavelet_setting["level"]
                                            )


        ret = {'origin_input': origin_inp,
               'input': inp,
               'input_lengths': input_len,
               'origin_corrupted_input': origin_corrupted_inp,
               'corrupted_input': corrupted_inp
               }

        return ret, {}