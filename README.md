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
3.Install mmcv-full 1.1.5
```
pip install mmcv-full== 1.1.5
```
4.Install build requirements
```
pip install -r mmdetection-master\requirements\build.txt
pip install -v -e .
```
