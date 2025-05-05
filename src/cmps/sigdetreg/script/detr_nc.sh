#!/bin/bash
# Test DETR with Conformer backbone

OUTPUT_DIR="exps/detr_nc"

cd .. && \
python src/pretrain.py --output_dir ${OUTPUT_DIR}/cmp --dataset ROD --epochs 400 --lr_drop 200 --model detr --backbone resnet18 --task sigdet --finetune --finetune_ratio 0.1 --batch_size 32 --no_cls && \
python src/pretrain.py --output_dir ${OUTPUT_DIR}/pretrain --dataset ROD --epochs 100 --lr_drop 50 --model detr --backbone resnet18 --task selfdet --object_embedding_loss --batch_size 32 --no_cls && \
python src/pretrain.py --output_dir ${OUTPUT_DIR}/finetune --dataset ROD --epochs 400 --lr_drop 200 --model detr --backbone resnet18 --task sigdet --pretrain ${OUTPUT_DIR}/pretrain/checkpoint.pth --finetune --finetune_ratio 0.1 --batch_size 32 --no_cls
