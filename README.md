# PyTorch-ENet for CamVid (11-Class)

A PyTorch implementation of ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation, with fixes for modern PyTorch and CamVid 11-class dataset.

## Overview

This repository is a fork of [davidtvs/PyTorch-ENet](https://github.com/davidtvs/PyTorch-ENet) with significant updates to make it compatible with:
- Modern PyTorch (2.x)
- CamVid dataset reduced to **11 semantic classes + unlabeled**
- Windows environment
- Single-channel grayscale labels (class indices)

## Key Fixes Applied

- Fixed label shape issues (`[B, 3, H, W]` → `[B, H, W]`)
- Updated `LongTensorToRGBPIL` for proper visualization
- Fixed `.next()` iterator deprecation (`iter().next()` → `next(iter())`)
- Added proper `ignore_index=255` handling for unlabeled pixels
- Resolved class_weights mismatch between original 32-class and 11-class setup
- Improved checkpoint loading for PyTorch 2.6+
- Fixed multiprocessing issues on Windows (`num_workers=0`)

## Requirements

- Python 3.8+
- PyTorch 2.x (CPU version)
- torchvision
- Pillow
- NumPy
- Matplotlib

## Dataset Setup (CamVid 11-Class)

1. Place your CamVid dataset in `datasets/CamVid/`
2. Folder structure should be:
datasets/CamVid/
├── train/
├── val/
├── test/
├── train_labels/     ← grayscale labels (values 0-10 + 255)
├── val_labels/
└── test_labels/
text**Note**: Labels must be single-channel grayscale images where pixel values represent class indices.

## Training

```bash
python main.py -m train \
 --save-dir save/ENet_CamVid_train \
 --name ENet_CamVid \
 --dataset camvid \
 --dataset-dir datasets/CamVid/ \
 --batch-size 8 \
 --epochs 200 \
 --learning-rate 0.0005 \
 --workers 0
Testing
Bashpython main.py -m test \
    --save-dir save/ENet_CamVid_train \
    --name ENet_CamVid \
    --dataset camvid \
    --dataset-dir datasets/CamVid/ \
    --batch-size 8 \
    --workers 0
Features

Real-time semantic segmentation using ENet
Support for CamVid 11-class + unlabeled
Proper handling of void/unlabeled pixels (255)
Visualization of predictions
Checkpoint saving and resuming

Original Repository
Forked from: davidtvs/PyTorch-ENet