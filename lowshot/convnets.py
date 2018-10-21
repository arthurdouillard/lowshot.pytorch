import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, blocks):
        super().__init__()

        self._convnet = []
        for block in blocks:
            self._convnet.append(
                nn.Conv2d(
                    block['in_channels'],
                    block['out_channels'],
                    kernel_size=block.get('kernel_size', 3),
                    stride=block.get('stride', 1),
                    padding=block.get('padding', 1),
                    bias=False
                )
            )
            self._convnet.append(nn.BatchNorm2d(block['out_channels']))

            if block['pooling']:
                self._convnet.append(nn.MaxPool2d(
                    block.get('pool_kernel_size', 2),
                    block.get('pool_stride'),
                    block.get('pool_padding', 0)
                ))

            if block.get('relu', False):
                self._convnet.append(nn.ReLU())

        self._convnet = nn.Sequential(*self._convnet)


    def forward(self, x):
        return self._convnet(x)
