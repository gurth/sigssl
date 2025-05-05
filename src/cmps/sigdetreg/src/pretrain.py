import torch
from torch.utils.data import DataLoader
import os
import numpy as np
import random
from pathlib import Path
import time

from args import args

from datasets.dataset_factory import get_dataset

from models import create_model, load_model, save_model
from models.backbones.backbone_factory import get_backbone, get_pure_backbone
from trains.train_factory import train_factory
from tests import evaluate, viz
from utils import misc

from utils.logger import Logger

PRETRAINING_DATASETS = []
PRINT_PARAMS = False

def pretrain(arg):
    if arg.frozen_weights is not None:
        assert arg.masks, "Frozen training is meant for segmentation only"

    device = torch.device(arg.device)

    seed = arg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Using random seed: {seed}")
    swav_model = None

    if arg.task == 'selfdet':
        swav_model = get_pure_backbone(arg)
        arg.pretrain_dir = "exps/backbone"
        backbone_path = os.path.join(arg.pretrain_dir, 'backbone.pth')
        if os.path.exists(backbone_path):
            backbone_checkpoint = torch.load(backbone_path)
            swav_model.load_state_dict(backbone_checkpoint)
            swav_model = swav_model.to(device)
            print('Loaded the backbone from the pre-trained model')

    Dataset = get_dataset(arg.dataset, arg.task)
    arg = args.update_dataset_info(arg, Dataset)

    logger = Logger(arg)

    model, criterion, postprocessors = create_model(arg)
    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = Dataset(arg, 'train')
    dataset_val = Dataset(arg, 'val')
    if arg.task == 'selfdet':
        cache_dir = os.path.join(arg.output_dir, 'cache')
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        dataset_train.init_selfdet(cache_dir=cache_dir, max_prop=30, strategy='topk')
        dataset_val.init_selfdet(cache_dir=cache_dir, max_prop=30, strategy='topk')

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, arg.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=misc.collate_fn, num_workers=arg.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, arg.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=misc.collate_fn, num_workers=arg.num_workers,
                                 pin_memory=True)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    if PRINT_PARAMS:
        for n, p in model.named_parameters():
            print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                 if not match_name_keywords(n, arg.lr_backbone_names) and not
                 match_name_keywords(n, arg.lr_linear_proj_names) and p.requires_grad],
            "lr": arg.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       match_name_keywords(n, arg.lr_backbone_names) and p.requires_grad],
            "lr": arg.lr_backbone,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       match_name_keywords(n, arg.lr_linear_proj_names) and p.requires_grad],
            "lr": arg.lr * arg.lr_linear_proj_mult,
        }
    ]

    if arg.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=arg.lr, momentum=0.9,
                                    weight_decay=arg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=arg.lr,
                                      weight_decay=arg.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,  step_size=arg.lr_drop)

    base_ds = dataset_val

    if arg.frozen_weights is not None:
        checkpoint = torch.load(arg.frozen_weights, map_location='cpu')
        model.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(arg.output_dir)
    if arg.pretrain:
        print('Initialized from the pre-training model')
        checkpoint = torch.load(arg.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            # remove useless class embed
            if 'class_embed' in k:
                del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)

    if arg.resume:
        model, optimizer, dlr_scheduler = load_model(model, optimizer, lr_scheduler, arg)
        backbone = model.backbone[0].body
        torch.save(backbone.state_dict(), os.path.join(arg.output_dir, 'backbone.pth'))


    if arg.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, arg.output_dir)

        if arg.output_dir:
            misc.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    if arg.viz:
        viz(model, criterion, postprocessors,
            data_loader_val, base_ds, device, arg.output_dir)
        return

    print("Start training")
    start_time = time.time()

    trainer = train_factory[arg.task](arg, model, criterion,
                                      postprocessors, base_ds, swav_model, optimizer,
                                      max_norm=arg.clip_max_norm)

    if arg.resume:
        val_stats = trainer.run_epoch("val", arg.start_epoch, data_loader_val)
        logger.log.write("<<<<<<[RUSUME]>>>>>>\n")
        logger.write_epoch(val_stats, arg.start_epoch, "val")



    for epoch in range(arg.start_epoch, arg.epochs):
        train_stats = trainer.run_epoch("train", epoch, data_loader_train)
        logger.write_epoch(train_stats, epoch, "train")
        lr_scheduler.step()

        # Debug: Print lr
        lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        print(f"Epoch {epoch} finished. LR: {lrs}")

        if arg.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']

            if (epoch + 1) % arg.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')

                val_stats = trainer.run_epoch("val", epoch, data_loader_val)
                logger.write_epoch(val_stats, epoch, "val")

            for checkpoint_path in checkpoint_paths:
                misc.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': arg,
                }, checkpoint_path)




if __name__ == '__main__':
    arg = args()
    arg= arg.parse()
    pretrain(arg)


