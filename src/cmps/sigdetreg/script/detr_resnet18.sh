#!/bin/bash
# Test DETR with Resnet18 backbone

OUTPUT_DIR="exps/test_detr_resnet18"

# If $1 is not empty, then resume from the checkpoint
if [ ! -z "$1" ]; then
    RESUME="--resume ${OUTPUT_DIR}/checkpoint.pth"
fi


conda activate detreg

cd .. && python src/pretrain.py ${RESUME} --output_dir ${OUTPUT_DIR} --dataset ROD --epochs 400 --lr_drop 400 --model detr --batch_size 16

