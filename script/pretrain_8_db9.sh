#!/bin/bash

python train.py selfdet --arch conformer_tiny --wt_decomp --no_cls --dbg_wt db9 --exp_id wt_nc_8_db9_pretrain --num_epochs 50 --val_intervals 10