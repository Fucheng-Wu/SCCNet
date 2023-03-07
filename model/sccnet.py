"""
original code from research:
https://github.com/Fucheng-Wu/SCCNet
"""

import torch
import torch.nn as nn

from channelcalibration import ChannelCalibrationBranch
from spatialcalibration import SpatialCalibrationBranch
from discretecosinetransform import DiscreteCosineTransformBlock


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    """ SCCNet Block"""
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim * 4)
        self.dwconv = nn.Conv2d(dim * 4, 4 * dim, kernel_size=3, stride=1, padding=1, groups=dim * 4)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.DCT = DiscreteCosineTransformBlock(dim * 4)
        self.CC = ChannelCalibrationBranch(dim * 4)
        self.SC = SpatialCalibrationBranch(dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = self.conv1(x)
        x = self.norm(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.DCT(x)
        x1 = self.CC(x)
        x2 = self.SC(x)
        x = x1 + x2
        x = self.conv2(x)
        # x = shortcut + self.drop_path(x)
        return x


class SCCNet(nn.Module):
    """ SCCNet"""
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem_downsampling = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                            nn.BatchNorm2d(dims[0]))
        self.downsample_layers.append(stem_downsampling)
        for i in range(3):
            downsample_layer = nn.Sequential(nn.BatchNorm2d(dims[i]),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x.mean([-2, -1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


def sccnet(num_classes: int):
    model = SCCNet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes)

    return model