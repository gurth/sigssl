import torch

import copy

from .backbones.backbone_factory import get_backbone
from .backbones.utils import Joiner

from .networks.def_detr.deformable_detr import DeformableDETR, SetCriterion as DefSetCriterion, PostProcess as DefPostProcess
from .networks.detr.detr import DETR, SetCriterion as DETRSetCriterion, PostProcess as DETRPostProcess
from .networks.def_detr.def_matcher import build_matcher as build_def_matcher
from .networks.detr.detr_matcher import build_matcher as build_detr_matcher
from .networks.detr_utils.segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .networks.detr_utils.deformable_transformer import build_deforamble_transformer

from .networks.detr_utils.transformer import build_transformer

from .networks.get_detr import get_detr
from .networks.get_def_detr import get_def_detr

_model_factory = {
    'detr': get_detr,
    'deformable_detr': get_def_detr,
}

def build_backbone(arg):
    model = get_backbone(arg)
    return model

def create_model(arg):
    num_classes = arg.num_classes

    device = torch.device(arg.device)

    weight_dict = {'loss_ce': arg.cls_loss_coef, 'loss_bbox': arg.bbox_loss_coef,
                   'loss_giou': arg.giou_loss_coef}
    if arg.masks:
        weight_dict["loss_mask"] = arg.mask_loss_coef
        weight_dict["loss_dice"] = arg.dice_loss_coef

    if arg.aux_loss:
        aux_weight_dict = {}
        for i in range(arg.dec_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in weight_dict.items()})

        # only in def detr impl.
        if arg.model == 'deformable_detr':
            aux_weight_dict.update(
                {k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    if arg.object_embedding_loss:
        losses.append('object_embedding')
        weight_dict['loss_object_embedding'] = arg.object_embedding_coef

    if arg.masks:
        losses += ["masks"]

    backbone = build_backbone(arg)

    if arg.model == 'deformable_detr':
        transformer = build_deforamble_transformer(arg)
        model = DeformableDETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=arg.num_queries,
            num_feature_levels=arg.num_feature_levels,
            aux_loss=arg.aux_loss,
            with_box_refine=arg.with_box_refine,
            two_stage=arg.two_stage,
            object_embedding_loss=arg.object_embedding_loss,
            obj_embedding_head=arg.obj_embedding_head
        )
        matcher = build_def_matcher(arg)
        criterion = DefSetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=arg.focal_alpha)
        postprocessors = {'bbox': DefPostProcess()}

    elif arg.model == 'detr':
        transformer = build_transformer(arg)
        model = DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=arg.num_queries,
            aux_loss=arg.aux_loss,
            object_embedding_loss=arg.object_embedding_loss,
            obj_embedding_head=arg.obj_embedding_head
        )
        matcher = build_detr_matcher(arg)
        criterion = DETRSetCriterion(num_classes, matcher, weight_dict, arg.eos_coef,
                                     losses, object_embedding_loss=arg.object_embedding_loss)
        postprocessors = {'bbox': DETRPostProcess()}
    else:
        raise ValueError("Wrong model.")

    criterion.to(device)

    if arg.masks:
        model = DETRsegm(model, freeze_detr=(arg.frozen_weights is not None))

    if arg.masks:
        postprocessors['segm'] = PostProcessSegm()
        if arg.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(
                is_thing_map, threshold=0.85)

    return model, criterion, postprocessors


def load_model(model, optimizer, lr_scheduler, arg):
    checkpoint = torch.load(arg.resume, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]

    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))

    if not arg.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        p_groups = copy.deepcopy(optimizer.param_groups)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for pg, pg_old in zip(optimizer.param_groups, p_groups):
            pg['lr'] = pg_old['lr']
            pg['initial_lr'] = pg_old['initial_lr']
        # print(optimizer.param_groups)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        # Hack for resuming from checkpoint and modifying lr scheduler
        arg.override_resumed_lr_drop = True
        if arg.override_resumed_lr_drop:
            print(
                'Warning: (hack) arg.override_resumed_lr_drop is set to True, so arg.lr_drop would override lr_drop in resumed lr_scheduler.')
            lr_scheduler.step_size = arg.lr_drop
            lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))

        lr_scheduler._step_count = lr_scheduler.last_epoch
        arg.start_epoch = checkpoint['epoch'] + 1


    return model, optimizer, lr_scheduler

def save_model(model):
    pass