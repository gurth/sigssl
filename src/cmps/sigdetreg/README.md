# DETR & Deformable DETR for Self-Supervised Learning in Time-Domain Radio Signal Detection

This repository provides modified implementations of **DETR** and **Deformable DETR** adapted for self-supervised learning in time-domain radio signal detection, as proposed in our paper.

Based on [DETReg](https://github.com/amirbar/DETReg).

---

## Installation

The setup is the same as in the main project.

### Additional Requirements

To install any additional dependencies specific to this module, run:

  ```shell
  pip install -r requirements.txt
  ```

### Compiling CUDA operators

```shell
cd ./src/models/networks/detr_utils/ops
sh ./make.sh
```


## Example

### Dataset preparation:

Please organize your dataset following the same structure as described in the main project's `README.md`, relative to this project root.

### Pre-training DETR:

```shell
python src/pretrain.py \
    --output_dir exps/detr_nc/pretrain \
    --dataset ROD \
    --epochs 100 \
    --lr_drop 50 \
    --model detr \
    --backbone resnet18 \
    --task selfdet \
    --object_embedding_loss \
    --batch_size 32 \
    --no_cls
```

### Fine-tuning DETR:

```shell
python src/pretrain.py \
    --output_dir exps/detr_nc/finetune \
    --dataset ROD \
    --epochs 400 \
    --lr_drop 200 \
    --model detr \
    --backbone resnet18 \
    --task sigdet \
    --pretrain exps/detr_nc/pretrain/checkpoint.pth \
    --finetune \
    --finetune_ratio 0.1 \
    --batch_size 32 \
    --no_cls
```

### Pre-training Deformable DETR:

```shell
python src/pretrain.py \
    --output_dir exps/def_nc/pretrain \
    --dataset ROD \
    --epochs 100 \
    --lr_drop 50 \
    --model deformable_detr \
    --backbone resnet18 \
    --task selfdet \
    --object_embedding_loss \
    --batch_size 32 \
    --no_cls
```

### Fine-tuning Deformable DETR:

```shell
python src/pretrain.py \
    --output_dir exps/def_nc/finetune \
    --dataset ROD \
    --epochs 400 \
    --lr_drop 200 \
    --model deformable_detr \
    --backbone resnet18 \
    --task sigdet \
    --pretrain exps/def_nc/pretrain/checkpoint.pth \
    --finetune \
    --finetune_ratio 0.1 \
    --batch_size 32 \
    --no_cls
```

## License

This code inherits the license of the main project. See the main `LICENSE` file for details.
