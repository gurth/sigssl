import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model


class BaseDetector(object):
    def __init__(self, opt):
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt.wavelet_setting,  )
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        # self.mean = np.array(opt.mean, dtype=np.float32)
        # self.std = np.array(opt.std, dtype=np.float32)

        self.opt = opt

    def process(self, s, return_time=False):
        raise NotImplementedError

    def pre_process(self, s):
        raise NotImplementedError

    def post_process(self, dets, meta):
        raise NotImplementedError

    def run(self, path_or_data):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        tot_time = 0
        meta = {'input_len': 0, 'output_len': 0}
        if self.opt.nms:
            meta['nms_setting'] = self.opt.nms_setting

        start_time = time.time()

        if isinstance(path_or_data, np.ndarray):
            data = path_or_data
        elif isinstance(path_or_data, str):
            if path_or_data.endswith('.npy'):
                data = np.load(path_or_data)
            elif path_or_data.endswith('.bin'):
                data = np.fromfile(path_or_data, dtype=np.float32)
                data = data.reshape(-1, 2)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        meta['input_len'] = data.shape[0]

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        data = self.pre_process(data)
        data = torch.from_numpy(data).unsqueeze(0)
        data = data.to(self.opt.device)

        torch.cuda.synchronize()
        pre_process_time = time.time()

        output, dets, forward_time = self.process(data, return_time=True)
        torch.cuda.synchronize()

        meta['output_len'] = output['hm'].shape[2]

        net_time += forward_time - pre_process_time

        decode_time = time.time()
        dec_time += decode_time - forward_time

        dets = self.post_process(dets, meta)

        post_process_time = time.time()
        post_time += post_process_time - decode_time

        end_time = time.time()
        tot_time += end_time - start_time

        return {'dets': dets, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time, 'dec': dec_time,
                'post': post_time}


