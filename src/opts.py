import argparse
import os
import sys


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('task', default='ctdet',
                                 help='ctdet')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--dataset', default='ROD',
                                 help='ROD')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed', type=int, default=42,
                                 help='random seed')
        self.parser.add_argument('--wt_decomp', action='store_true',
                                 help='use wavelets decomposition')

        # model
        self.parser.add_argument('--down_ratio', type=int, default=1,
                                 help='')
        self.parser.add_argument('--no_cls', action='store_true',
                                 help='not predict classes')

        # dataset
        self.parser.add_argument('--ext', action='store_true',
                                 help='extern dataset for low SNR')

        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='90,120',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=140,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=4,
                                 help='batch size')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                 help='include validation in training and '
                                      'test on test set')

        # test
        self.parser.add_argument('--K', type=int, default=100,
                                 help='max number of output objects.')
        self.parser.add_argument('--val_cls', action='store_true',
                                 help='eval on class ap')
        self.parser.add_argument('--not_nms', action='store_true',
                                 help='not use nms.')
        self.parser.add_argument('--denoiser', default='wt_thresh',
                                 help='denoiser type')

        # log
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        # TODO: Find the best threshold
        self.parser.add_argument('--vis_thresh', type=float, default=0,
                                 help='visualization threshold.')
        # | v | n | Pf | Pd |
        # |0.06 | 0.3 | 0.9909 | 0.0971 |
        # |0.058 | 0.3 | 0.9910 |  0.0999
        #
        # |0.5 | 0.3 | FPS 6.3455 |AP50 0.8607 |AP75 0.8514 |AP95 0.7567 |Pd 0.9096 |Pf 0.0465


        # conformer
        self.parser.add_argument('--arch', default='conformer_small',
                                 help='${model architecture}_${scale}')
        self.parser.add_argument('--frame_length', type=int, default=32,
                                 help='length of each patch')

        # ctdet
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wt_weight', type=float, default=0.1,
                                 help='loss weight for bounding box size.')
        self.parser.add_argument('--not_absolute_offset', action='store_true',
                                 help='predict absolute offset.')
        self.parser.add_argument('--not_reg_offset', action='store_true',
                                 help='not predict absolute offset.')
        self.parser.add_argument('--head_conv', type=int, default=64,
                                 help='conv layer channels for output head')

        # Selfdet
        self.parser.add_argument('--selfdet_strategy', default='topk', type=str, help='strategy to select proposals')
        self.parser.add_argument('--load_dino', action='store_true', help='load dino pretrained model')
        self.parser.add_argument('--drop_path_rate', type=float, default=0,
                                 help='drop path rate for student model')

        # EISS
        self.parser.add_argument('--eiss_strategy', default='noise', type=str, help='strategy to corrupt the data')
        self.parser.add_argument('--eiss_alpha', default=0.1, type=float, help='alpha value for eiss')
        self.parser.add_argument('--eiss_num_steps', default=2, type=int, help='step for eiss')
        
        # TODO：debug, should delete
        self.parser.add_argument('--dbg_wt', default='db4', help='debug wavelet type')
        self.parser.add_argument('--dbg_wt_level', type=int,  default=1, help='debug wavelet type')
        self.parser.add_argument('--dbg_best', action='store_true', help='load best model')

        self.parser.add_argument("--finetune_ratio", type=float, default=-1, help="finetune ratio")
        self.parser.add_argument('--finetune', action='store_true', help='finetune the model')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
            opt.no_arg = True
        else:
            opt = self.parser.parse_args(args)
            opt.no_arg = False

        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        opt.val_cls = False if opt.no_cls else opt.val_cls
        # How about do not use not_?
        opt.reg_offset = not opt.not_reg_offset
        opt.not_val_cls = not opt.val_cls
        opt.nms = not opt.not_nms

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
        opt.data_dir = os.path.join(opt.root_dir, 'data')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        if not opt.no_arg:
            print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth') if not opt.dbg_best \
                else os.path.join(model_path, 'model_best.pth')

        if opt.wt_decomp:
            opt.wavelet_setting = {
                'wavelet': 'db4',
                'level': 1
            }
            try:
                # import pycudwt
                # opt.wavelet_setting["cuda"] = True
                # TODO: Implement CUDA wavelet transform
                raise NotImplementedError
            except:
                opt.wavelet_setting["cuda"] = False
        else:
            opt.wavelet_setting = None

        opt.nms_setting = {
            'Nt': 0.3,
            'sigma': 0.5,
            'thresh': 0.001,
            'method': 'gaussian'
        } if opt.nms else None

        if opt.wt_decomp:
            opt.wavelet_setting["wavelet"] = opt.dbg_wt
            opt.wavelet_setting["level"] = opt.dbg_wt_level

        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes
        opt.class_name = dataset.class_name

        if opt.task == 'ctdet' or opt.task == 'selfdet':
            assert opt.dataset in ['ROD']
            opt.heads = {'hm': 1 if opt.no_cls else opt.num_classes,
                         'wt': 1}
            if opt.reg_offset:
                opt.heads.update({'reg': 1})
        elif opt.task == 'eiss':
            assert opt.dataset in ['ROD']
            opt.heads = {"ei": 1}
        elif opt.task == 'simclr':
            assert opt.dataset in ['ROD']
            opt.heads = {"simclr": 1}
        elif opt.task == 'TC':
            assert opt.dataset in ['ROD']
            opt.heads = {}
        else:
            if not opt.no_arg:
                assert 0, 'task not defined!'

        if not opt.no_arg:
            print('heads', opt.heads)
        return opt
