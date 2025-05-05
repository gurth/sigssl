import argparse
import os
import sys
from pathlib import Path
import numpy as np

default_settings = {
    'deformable_detr': {
        'lr': 2e-4,
        'lr_backbone': 2e-5,
        'epochs': 50,
        'lr_drop': 40,
        'dim_feedforward': 1024,
        'num_queries': 300,
        'set_cost_class': 2,
        'cls_loss_coef': 2
    },
    'detr': {
        'lr': 1e-4,
        'lr_backbone': 1e-5,
        'epochs': 300,
        'lr_drop': 200,
        'dim_feedforward': 256,
        'num_queries': 100,
        'set_cost_class': 1,
        'cls_loss_coef': 1
    }
}

def set_model_defaults(args):
    defaults = default_settings[args.model]
    runtime_args = vars(args)
    for k, v in runtime_args.items():
        if v is None and k in defaults:
            setattr(args, k, defaults[k])
    return args

def set_dataset_path(args):
    args.rod_path = os.path.join(args.data_root, 'rod')

class args(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--model', default='detr', type=str, choices=['detr', 'deformable_detr'])
        parser.add_argument('--device', default='cuda',
                            help='device to use for training / testing')
        parser.add_argument('--task', default='sigdet', help='task')
        parser.add_argument('--no_cls', action='store_true', help='no classification')

        parser.add_argument('--lr', type=float)
        parser.add_argument('--max_prop', default=30, type=int)
        # parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
        # parser.add_argument('--lr_backbone', type=float)
        parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str,
                            nargs='+')
        parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--weight_decay', default=1e-4, type=float)
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--lr_drop', type=int)
        parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
        parser.add_argument('--clip_max_norm', default=0.1, type=float,
                            help='gradient clipping max norm')
        parser.add_argument('--sgd', action='store_true')



        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--viz', action='store_true')
        parser.add_argument('--num_workers', default=2, type=int)
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')

        # Dataset parameters
        parser.add_argument('--dataset', default='ROD')
        parser.add_argument('--data_root', default='data')
        parser.add_argument('--output_dir', default='',
                                 help='path where to save, empty for no saving')
        parser.add_argument('--cache_path', default=None, help='where to store the cache')
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--resume', default='', help='resume from checkpoint')
        parser.add_argument('--random_seed', action='store_true')
        parser.add_argument('--ext', action='store_true', help='use extra data')

        parser.add_argument('--pretrain', default='', help='initialized from the pre-training model')

        parser.add_argument('--finetune', action='store_true', help='finetune the model')
        parser.add_argument("--finetune_ratio", type=float, default=-1, help="finetune ratio")

        # Model parameters
        parser.add_argument('--frozen_weights', type=str, default=None,
                            help="Path to the pretrained model. If set, only the mask head will be trained")

        # * Backbone
        parser.add_argument('--backbone', default='resnet18', type=str,
                            help="Name of the convolutional backbone to use")
        parser.add_argument('--load_backbone', default='', type=str)
        parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
        parser.add_argument('--lr_backbone', type=float)
        parser.add_argument('--dilation', action='store_true',
                            help="If true, we replace stride with dilation in the last convolutional block (DC5)")
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                            help="position / size * scale")

        parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

        # * Segmentation
        parser.add_argument('--masks', action='store_true',
                            help="Train segmentation head if the flag is provided")

        # * Matcher
        parser.add_argument('--set_cost_class', type=float,
                            help="Class coefficient in the matching cost")
        parser.add_argument('--set_cost_bbox', default=5, type=float,
                            help="L1 box coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=2, type=float,
                            help="giou box coefficient in the matching cost")
        parser.add_argument('--object_embedding_coef', default=1, type=float,
                            help="object_embedding_coef box coefficient in the matching cost")

        # * Loss coefficients
        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                            help="Disables auxiliary decoding losses (loss at each layer)")
        parser.add_argument('--mask_loss_coef', default=1, type=float)
        parser.add_argument('--dice_loss_coef', default=1, type=float)
        parser.add_argument('--cls_loss_coef', type=float)
        parser.add_argument('--bbox_loss_coef', default=5, type=float)
        parser.add_argument('--giou_loss_coef', default=2, type=float)
        parser.add_argument('--focal_alpha', default=0.25, type=float)

        parser.add_argument('--object_embedding_loss', default=False, action='store_true',
                            help='whether to use this loss')

        # * Transformer
        parser.add_argument('--enc_layers', default=4, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=4, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=128,  type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=128, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_queries', type=int, help="Number of query slots")
        parser.add_argument('--dec_n_points', default=4, type=int)
        parser.add_argument('--enc_n_points', default=4, type=int)

        # detr
        parser.add_argument('--obj_embedding_head', default='intermediate', type=str, choices=['intermediate', 'head'])
        parser.add_argument('--pre_norm', action='store_true')
        parser.add_argument('--eos_coef', default=0.1, type=float,
                            help="Relative classification weight of the no-object class")

        # Deformable DETR
        parser.add_argument('--with_box_refine', default=False, action='store_true')
        parser.add_argument('--two_stage', default=False, action='store_true')

        # Selfdet
        parser.add_argument('--strategy', default='ss_topk', type=str, help='strategy to select proposals')

        # val
        parser.add_argument('--save_result', action='store_true', help='save all det results')

        self.parser = parser


    def parse(self):
        args = self.parser.parse_args()
        set_dataset_path(args)
        args = set_model_defaults(args)

        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        # fix the seed for reproducibility
        if args.random_seed:
            args.seed = np.random.randint(0, 1000000)

        return args

    @staticmethod
    def update_dataset_info(opt, dataset):
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes
        opt.class_name = dataset.class_name

        return opt
