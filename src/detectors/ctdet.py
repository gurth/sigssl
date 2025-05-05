import numpy as np
from progress.bar import Bar
import time

import torch

from .base_detector import BaseDetector
from models.decode import ctdet_decode

from utils.radio import soft_nms
from utils.radio import wavelet_decomposition

class CtdetDetector(BaseDetector):
    def __init__(self, opt):
        super(CtdetDetector, self).__init__(opt)

    def pre_process(self, s):
        mean = [np.mean(s[:, 0]), np.mean(s[:, 1])]
        std = [np.std(s[:, 0]), np.std(s[:, 1])]
        s = (s - mean) / std

        # s = s.reshape(-1)
        # Reshape [131072] to [131072/64, 64]
        # s = s.reshape(-1, self.opt.frame_length)

        if self.opt.wt_decomp:

            if not self.opt.wavelet_setting["cuda"]:
                s = wavelet_decomposition(s,
                        self.opt.wavelet_setting["wavelet"],
                        self.opt.wavelet_setting["level"]
                )

        return s

    def process(self, s, return_time=False):

        with torch.no_grad():
            output = self.model(s, s.size()[1])
            hm = output['hm'].sigmoid_()
            # DEBUG ONLY
            hm_np = hm.detach().cpu().numpy()
            np.save("hm.npy", hm_np)

            wt = output['wt']
            reg = output['reg']

        torch.cuda.synchronize()
        forward_time = time.time()
        dets = ctdet_decode(hm, wt, reg=reg, K=self.opt.K)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

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
