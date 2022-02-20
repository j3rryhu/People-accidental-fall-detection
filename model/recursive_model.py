from model.network import make_block
import torch
from torch import nn


class BodyPoseModel(nn.Module):
    def __init__(self):
        super(BodyPoseModel, self).__init__()
        self.blocks = {}
        no_relu_layers = ['conv5_5_l1', 'conv5_5_l2', 'conv_stage2_7_l1',
                          'conv_stage2_7_l2', 'conv_stage3_7_l1', 'conv_stage3_7_l2',
                          'conv_stage4_7_l1', 'conv_stage4_7_l2', 'conv_stage5_7_l1',
                          'conv_stage5_7_l2', 'conv_stage6_7_l1', 'conv_stage6_7_l2']
        block0 = {
            'conv1_1': [3, 64, 3, 1, 1],
            'conv1_2': [64, 64, 3, 1, 1],
            'pool1': [2, 2, 0],
            'conv2_1': [64, 128, 3, 1, 1],
            'conv2_2': [128, 128, 3, 1, 1],
            'pool2': [2, 2, 0],
            'conv3_1': [128, 256, 3, 1, 1],
            'conv3_2': [256, 256, 3, 1, 1],
            'conv3_3': [256, 256, 3, 1, 1],
            'conv3_4': [256, 256, 3, 1, 1],
            'pool3': [2, 2, 0],
            'conv4_1': [256, 512, 3, 1, 1],
            'conv4_2': [512, 512, 3, 1, 1],
            'conv4_3': [512, 256, 3, 1, 1],
            'conv4_4': [256, 128, 3, 1, 1]
        }
        self.block0 = make_block(block0, no_relu_layers)
        block1_1 = {
            'conv5_1_l1': [128, 128, 3, 1, 1],
            'conv5_2_l1': [128, 128, 3, 1, 1],
            'conv5_3_l1': [128, 128, 3, 1, 1],
            'conv5_4_l1': [128, 512, 1, 1, 0],
            'conv5_5_l1': [512, 19, 1, 1, 0]
        }
        block1_2 = {
            'conv5_1_l2': [128, 128, 3, 1, 1],
            'conv5_2_l2': [128, 128, 3, 1, 1],
            'conv5_3_l2': [128, 128, 3, 1, 1],
            'conv5_4_l2': [128, 512, 1, 1, 0],
            'conv5_5_l2': [512, 38, 1, 1, 0]
        }
        self.block1_1 = make_block(block1_1, no_relu_layers)
        self.block1_2 = make_block(block1_2, no_relu_layers)
        self.blocks['block0'] = self.block0
        self.blocks['block1_1'] = self.block1_1
        self.blocks['block1_2'] = self.block1_2
        for i in range(2, 7):
            block_l1 = {
                'conv_stage{}_1_l1'.format(i): [185, 128, 7, 1, 3],
                'conv_stage{}_2_l1'.format(i): [128, 128, 7, 1, 3],
                'conv_stage{}_3_l1'.format(i): [128, 128, 7, 1, 3],
                'conv_stage{}_4_l1'.format(i): [128, 128, 7, 1, 3],
                'conv_stage{}_5_l1'.format(i): [128, 128, 7, 1, 3],
                'conv_stage{}_6_l1'.format(i): [128, 128, 1, 1, 0],
                'conv_stage{}_7_l1'.format(i): [128, 19, 1, 1, 0],
            }
            block_l2 = {
                'conv_stage{}_1_l2'.format(i): [185, 128, 7, 1, 3],
                'conv_stage{}_2_l2'.format(i): [128, 128, 7, 1, 3],
                'conv_stage{}_3_l2'.format(i): [128, 128, 7, 1, 3],
                'conv_stage{}_4_l2'.format(i): [128, 128, 7, 1, 3],
                'conv_stage{}_5_l2'.format(i): [128, 128, 7, 1, 3],
                'conv_stage{}_6_l2'.format(i): [128, 128, 1, 1, 0],
                'conv_stage{}_7_l2'.format(i): [128, 38, 1, 1, 0],
            }
            self.blocks['block{}_1'.format(i)] = make_block(block_l1, no_relu_layers)
            self.blocks['block{}_2'.format(i)] = make_block(block_l2, no_relu_layers)
        self.block2_1 = self.blocks['block2_1']
        self.block2_2 = self.blocks['block2_2']
        self.block3_1 = self.blocks['block3_1']
        self.block3_2 = self.blocks['block3_2']
        self.block4_1 = self.blocks['block4_1']
        self.block4_2 = self.blocks['block4_2']
        self.block5_1 = self.blocks['block5_1']
        self.block5_2 = self.blocks['block5_2']
        self.block6_1 = self.blocks['block6_1']
        self.block6_2 = self.blocks['block6_2']

    def forward(self, x):
        x = self.block0(x)
        for i in range(1, 6):
            l1 = self.blocks['block{}_1'.format(i)](x)
            l2 = self.blocks['block{}_2'.format(i)](x)
            x = torch.cat([l1, l2, x], 1)
        l1 = self.block6_1(x)
        l2 = self.block6_2(x)
        return l1, l2


