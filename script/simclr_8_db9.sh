#!/usr/bin/env bash

cd .. && python src/train.py simclr --arch conformer_tiny --wt_decomp --no_cls --dbg_wt db9 --exp_id test_simclr --num_epochs 50 --val_intervals 5 --batch_size 4