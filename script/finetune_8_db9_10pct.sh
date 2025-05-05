#!/usr/bin/env bash

cd .. && python src/train.py ctdet --finetune --finetune_ratio 0.1 --arch conformer_tiny --wt_decomp --no_cls --dbg_wt db9 --exp_id wt_nc_8_db9_finetune  --resume  --num_epochs 400 --val_intervals 20