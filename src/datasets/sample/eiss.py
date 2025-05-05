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

class EISSDataset(data.Dataset):
    def init_eiss(self, augmenter_strategy="noise"):
        self.augmenter = augmenter_factory[augmenter_strategy]()

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

        origin_corrupted_inp = torch.tensor(self.augmenter(origin_inp), dtype=torch.float32)
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