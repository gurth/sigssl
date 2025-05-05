#!/bin/bash

conda activate dl
cd .. && python src/train.py TC --arch conformer_tiny --wt_decomp --no_cls --dbg_wt db9 --exp_id test_TC --num_epochs 100 --val_intervals 20 --batch_size 8