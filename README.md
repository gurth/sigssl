# Implementation of the Paper: *Self-Supervised Learning for Time-Domain Radio Signal Detection*

Chengzhi Ji and Xin Zhou\*

---

## Introduction

Time-domain radio signal detection is critical for precise signal analysis. Although advanced deep learning methods excel in complex scenarios, their performance is often constrained by the limited availability of labeled data. In this paper, we propose a self-supervised framework for time-domain radio signal detection, along with a novel signal region proposal algorithm, Signal Selective Search. Experiments on various datasets demonstrate that self-supervised learning improves detection performance in data-scarce scenarios, achieving nearly a 20-point average precision (AP) increase with only 50 labeled real-world samples. Additional unlabeled pre-training further enhances results, highlighting the practical value of self-supervised pre-training in radio signal detection.

## Installation

### Requirements

- Linux with CUDA >= 12.1
- Python >= 3.10

### Setup

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

* Alternatively, you can run the code in your own environment by ensuring all necessary dependencies are installed.

## Example

### Dataset preparation:

Please organize your I/Q sample data and annotations  in the following directory structure:

```
code_root/
└── data/
    ├── mod_bin/
          ├── 00001.npy
          ├── 00002.npy
          └── ...
    └── mod_lab/
          ├── 00001.npy
          ├── 00002.npy
          └── ...
```

* Each `.npy` file in `mod_bin` represents an I/Q sample with shape `[Length, 2]`.
* Each corresponding `.npy` file in `mod_lab` contains annotations with the following structure:

```
[{
'bbox': [FREQ_BEG, TIME_BEG, FREQ_END, TIME_END], 
'category': MODULATION_TYPE, 
'property': [TOTAL_TIME, SAMPLE_RATE, SYMBOL_RATE, POWER]
},
...]
```

### Pre-training the ConforDet Model:

```bash
python src/train.py selfdet \
    --arch conformer_tiny \
    --wt_decomp \
    --no_cls \
    --dbg_wt db9 \
    --exp_id wt_nc_8_db9_pretrain \
    --num_epochs 50 \
    --val_intervals 10
```

### Fine-tuning the ConforDet Model:

```bash
python src/train.py ctdet \
    --finetune \
    --finetune_ratio 0.1 \
    --arch conformer_tiny \
    --wt_decomp \
    --no_cls \
    --dbg_wt db9 \
    --exp_id wt_nc_8_db9_finetune \
    --resume \
    --num_epochs 400 \
    --val_intervals 20
```

### Additional Comparisons：

We extend **DETR** and **Deformable DETR** for time-domain signal detection. Their implementations can be found in: `src/cmps/sigdetreg`. Please refer to the `README.md` in that directory for detailed usage.

You can also pre-train using other self-supervised methods:

SimCLR:

```shell
python src/train.py simclr \
    --arch conformer_tiny \
    --wt_decomp \
    --no_cls \
    --dbg_wt db9 \
    --exp_id wt_nc_8_db9_simclr \
    --num_epochs 50 \
    --val_intervals 5 \
    --batch_size 4
```

TS-TCC:

``` shell
python src/train.py TC \
    --arch conformer_tiny \
    --wt_decomp \
    --no_cls \
    --dbg_wt db9 \
    --exp_id wt_nc_8_db9_TC \
    --num_epochs 100 \
    --val_intervals 20 \
    --batch_size 8
```

## Acknowlegments

We gratefully acknowledge the contributions of the open-source projects referenced in our work.

## License

This project is released under the Apache 2.0 license. Please see the `LICENSE` file for more information.