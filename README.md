#  People's accidental fall detection (In progress)

## 1. Introduction

The application aims at giving alert on any accidental falls of people living by themselves using human pose estimation. The application uses algorithm from paper [Realtime Multi-Person 2D Pose Estimation using Part Affifinity Fields](https://arxiv.org/abs/1611.08050) to generate the human pose feature points. And calculate the angle between human body and floor to see whether one person is on the ground for more than five seconds.

## 2. Prerequisites

Python 3.7

PyTorch==1.7.1

Windows10 20H2

## 3. Requirements

Run `pip install -r requirements.txt`

## 4. Training

Put the WFLW dataset folder in the parent folder.  Run training.py which will generate a folder models which saves trained model in ckpt.pth.tar.

