# Object-detection

## Introduction

## Environment
* Windows
* CUDA 10.2
* cuDNN v8.0.2
* GPU: NVIDIA GeForce GTX 1660 Ti

## Installation
1. Create a conda environment
```
conda create -n mmdetection python=3.8 -y
conda activate mmdetection
```
2. Install PyTorch 1.6.0+cu101
```
pip install torch==1.6.0 torchvision==0.7.0
```
3. Install mmcv-full 1.1.5
```
pip install mmcv-full== 1.1.5
```
4. Download mmdetection at <https://github.com/open-mmlab/mmdetection>

5. Install build requirements
```
pip install -r mmdetection-master\requirements\build.txt
pip install -v -e .
```
## Data
1. You need to download WIDER Face Training, Vlidation, Testing Image from <http://shuoyang1213.me/WIDERFACE/>
2. Run ``` convert.py ``` to split images and annotations
3. Create folder ```data``` in ``` mmdetection-master```
Directory shold like this:
```
-- data
   |-- WIDERFace
       |-- WIDER_train
       |   |-- 0--parade
       |   |-- 1--Handshaking
       |   ...
       |   |-- annotations
       |-- WIDER_val
       |   |-- 0--parade
       |   |-- 1--Handshaking
       |   ...
       |   |-- annotations
       |-- train.txt
       |-- val.txt
```
## Train
1. Modify ```configs\yolo\yolov3_d53_mstrain-608_273e_coco_v1.py```
Replace the Coco data set with Widerface
```
# dataset settings
dataset_type = 'WIDERFaceDataset'
data_root = 'data/WIDERFace/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(608, 608), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'train.txt',
            img_prefix=data_root + 'WIDER_train/',
            min_size=17,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root + 'WIDER_val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'val.txt',
        img_prefix=data_root + 'WIDER_val/',
        pipeline=test_pipeline))
```
