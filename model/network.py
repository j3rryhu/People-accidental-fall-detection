import torch
import torch.nn as nn
from collections import OrderedDict


def make_block(layers, no_relu_layers):
    block = []
    for name, attr in layers.items():
        if 'conv' in name:
            if name in no_relu_layers:
                conv = nn.Conv2d(in_channels=attr[0], out_channels=attr[1], kernel_size=attr[2], stride=attr[3], padding=attr[4])
                block.append((name, conv))
            else:
                conv = nn.Conv2d(in_channels=attr[0], out_channels=attr[1], kernel_size=attr[2], stride=attr[3], padding=attr[4])
                relu = nn.ReLU()
                block.append((name, conv))
                block.append(('relu_'+name, relu))
        if 'pool' in name:
            maxpool = nn.MaxPool2d(kernel_size=attr[0], stride=attr[1], padding=attr[2])
            block.append((name, maxpool))
    return nn.Sequential(OrderedDict(block))


