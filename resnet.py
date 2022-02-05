from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from kwinners import Kwinners2d


class FullBlock(nn.Module):
    def __init__(
        self, input_filters: int, output_filters: int, block_id: int, sparsity: float
    ):
        """Full Residual Block for ResNet

        Args:
            input_filters (int): number of input filters of the block
            output_filters (int): number of output filters of the block
            block_id (int): block id (for naming)
            sparsity (float): percent of non-zero activations.
        """
        super().__init__()
        k = max(4, int(output_filters * sparsity))
        self.pre_conv = nn.Conv2d(input_filters, output_filters, 3, padding=1, stride=1)
        self.pre_bn = nn.BatchNorm2d(output_filters)
        self.pre_relu = nn.ReLU()
        self.pre_kwin = Kwinners2d(k)

        self.conv1 = nn.Conv2d(output_filters, output_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_filters)
        self.relu1 = nn.ReLU()
        self.kwin1 = Kwinners2d(k)
        self.conv2 = nn.Conv2d(output_filters, output_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_filters)
        self.relu2 = nn.ReLU()
        self.kwin2 = Kwinners2d(k)

        self.maxpool = nn.MaxPool2d(2)
        self.res_conv = nn.Conv2d(output_filters, output_filters, 1, padding=0)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.pre_bn(x)
        x = self.pre_relu(x)
        res = self.res_conv(x)

        x = self.pre_kwin(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.kwin1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.kwin2(x)
        out = x + res
        out = self.maxpool(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        blocks: int,
        input_filters: int,
        num_classes: int = 10,
        sparsity: float = 1.0,
    ):
        """ResNet model class

        Args:
            blocks (int): number of blocks
            input_filters (int): number of input filters
            num_classes (int, optional): output vector size. Defaults to 10.
            sparsity (float, optional): percentage of non-zero activations. 
            Defaults to 1.0.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FullBlock(
                    input_filters * (2 ** i), input_filters * 2 * (2 ** i), i, sparsity
                )
                for i in range(blocks)
            ]
        )

        self.conv_input = nn.Conv2d(1, input_filters, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(input_filters)
        self.relu_input = nn.ReLU()
        self.kwin_input = Kwinners2d(max(4, int(sparsity * input_filters)))

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc0 = nn.Linear(input_filters * (2 ** blocks) * 14 * 14, 512)
        self.fc_out = nn.Linear(512, num_classes)

        self.bn_fc0 = nn.BatchNorm1d(512)

        self.relu_fc0 = nn.ReLU()

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        x = self.conv_input(x)
        x = self.bn_input(x)
        x = self.relu_input(x)
        x = self.kwin_input(x)

        for l in self.layers:
            x = l(x)

        out = x.view(x.size()[0], -1)
        out = self.fc0(out)
        out = self.bn_fc0(out)
        out = self.relu_fc0(out)

        out = self.fc_out(out)
        return out


def resnet(num_classes: int = 10, sparsity: float = 1.0) -> nn.Module:
    """creates a ResNet model

    Args:
        num_classes (int, optional): size of output vector. Defaults to 10.
        sparsity (float, optional): percentage of non-zero activations. 
        Defaults to 1.0.

    Returns:
        nn.Module: ResNet model
    """
    return ResNet(1, 64, num_classes, sparsity=sparsity)
