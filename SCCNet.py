"""
original code from facebook research:
https://github.com/facebookresearch/ConvNeXt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformer import TransformerEncoder
from layer import DCTLayer

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class DiscreteCosineTransform(nn.Module):
    def __init__(self, input_c):
        global _mapper_x, _mapper_y
        super(DiscreteCosineTransform, self).__init__()

        self.fc1 = nn.Conv2d(input_c, input_c, 1)
        self.fc2 = nn.Conv2d(input_c, input_c, 1)

        dct = dict([(3, 224), (384, 56), (768, 28), (1536, 14), (3072, 7)])
        self.dct = DCTLayer(input_c, dct[input_c], dct[input_c])

    def forward(self, x):
        scale = self.dct(x)
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)

        return scale * x

class ChannelCalibration(nn.Module):
    def __init__(self, input_c: int):
        super(ChannelCalibration, self).__init__()
        self.fc1 = nn.Conv2d(input_c, input_c, 1)
        self.fc2 = nn.Conv2d(input_c, input_c, 1, groups=input_c)


    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)

        return scale * x

class SpatialCalibration(nn.Module):
    # 将SE的全局池化改为对channel的mean
    def __init__(self,
                 input_c: int,
                 squeeze_factor: int = 4,
                 transformer_dim: int = 1,
                 ffn_latent_dim: int = 384,
                 # Transformer block的堆叠次数
                 n_transformer_blocks: int = 2,
                 # MSA中每个头的维度
                 head_dim: int = 1,
                 # Transformer Encoder中MSA内部的Dropout
                 attn_dropout: float = 0.0,
                 # Transformer Encoder中MSA block里Dropout的概率
                 dropout: float = 0.0,
                 # feed forward network中MLP内的Dropout概率
                 ffn_dropout: float = 0.0):
        super(SpatialCalibration, self).__init__()
        self.fc1 = nn.Conv2d(1, 1, 1)
        # self.fc2 = nn.Conv2d(squeeze_c, 1, 1)
        num_heads = transformer_dim // head_dim
        # feed forward network，也就是Transformer Encoder中MSA，模块之后的前馈模块
        self.transformer = TransformerEncoder(
            embed_dim=transformer_dim,
            ffn_latent_dim=ffn_latent_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        # x : [B,C,H,W]
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1) # x : [B,H,W,C]
        scale = x.mean(3, keepdim=True)
        scale = scale.view(B, -1, 1)
        # print(scale.shape) # [16, 784, 1]
        scale = self.transformer(scale)
        scale = scale.view(B, H, W, 1)
        scale = scale.permute(0, 3, 1, 2)  # [B,1,H,W]
        scale = self.fc1(scale)
        # scale = F.relu(scale, inplace=True)
        # scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        # x = x.permute(0, 3, 1, 2) # [B,C,H,W]

        return scale * shortcut

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=1)  # depthwise conv
        self.norm = nn.BatchNorm2d(dim * 4)
        self.dwconv = nn.Conv2d(dim * 4, 4 * dim, kernel_size=3, stride=1, padding=1, groups=dim * 4)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        # layer scale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        shortcut = x
        x = self.conv1(x)
        x = self.norm(x)
        # x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]

        x = self.dwconv(x)
        x = self.act(x)
        _, C, _, _ = x.shape

        DCT = DiscreteCosineTransform(C)
        DCT = DCT.to(device)
        x = DCT(x)

        CC = ChannelCalibration(C)
        CC = CC.to(device)
        x1 = CC(x)

        SC = SpatialCalibration(C)
        SC = SC.to(device)
        x2 = SC(x)

        x = x1 + x2
        x = self.conv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        # x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        # x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans: int = 3, num_classes: int = 1000, depths: list = None,
                 dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        # stem_downsampling为最初的下采样部分
        stem_downsampling = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                            nn.BatchNorm2d(dims[0]))
        self.downsample_layers.append(stem_downsampling)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(nn.BatchNorm2d(dims[i]),
                                             nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
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

        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


def SCCNet(num_classes: int):

    model = ConvNeXt(depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     num_classes=num_classes)
    return model