# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from typing import List
from torch import Tensor
import os
import copy
from mmcv.cnn import build_norm_layer
from math import log
import numpy
import matplotlib.pyplot as plt

try:
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


class DRFD(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super().__init__()
        out_c = channel * 2
        self.conv = nn.Conv2d(channel, channel * 2, kernel_size=3, stride=1, padding=1, groups=channel)
        self.conv_c = nn.Conv2d(channel * 2, channel * 2, kernel_size=3, stride=2, padding=1, groups=channel * 2)
        self.act_c = act_layer()
        self.norm_c = build_norm_layer(norm_layer, channel * 2)[1]
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = build_norm_layer(norm_layer, channel * 2)[1]
        self.fusion = nn.Conv2d(channel * 4, out_c, kernel_size=1, stride=1)
        # gaussian
        self.gaussian = Gaussian(out_c, 5, 0.5, norm_layer, act_layer)
        self.norm_g = build_norm_layer(norm_layer, out_c)[1]

    def forward(self, x):  # x = [B, C, H, W]

        x = self.conv(x)  # x = [B, 2C, H, W]
        gaussian = self.gaussian(x)
        x = self.norm_g(x + gaussian)
        max = self.norm_m(self.max_m(x))  # m = [B, 2C, H/2, W/2]
        conv = self.norm_c(self.act_c(self.conv_c(x)))  # c = [B, 2C, H/2, W/2]
        x = torch.cat([conv, max], dim=1)  # x = [B, 2C+2C, H/2, W/2]  -->  [B, 4C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 4C, H/2, W/2]     -->  [B, 2C, H/2, W/2]

        return x


class Conv_Extra(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Conv_Extra, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(channel, 64, 1),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
                                   build_norm_layer(norm_layer, 64)[1],
                                   act_layer(),
                                   nn.Conv2d(64, channel, 1),
                                   build_norm_layer(norm_layer, channel)[1])
    def forward(self, x):
        out = self.block(x)
        return out


class Scharr(nn.Module):
    def __init__(self, channel, norm_layer, act_layer):
        super(Scharr, self).__init__()
        # 定义Scharr滤波器
        scharr_x = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        scharr_y = torch.tensor([[-3., -10., -3.], [0., 0., 0.], [3., 10., 3.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.conv_x = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.conv_y = nn.Conv2d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        # 将Sobel滤波器分配给卷积层
        self.conv_x.weight.data = scharr_x.repeat(channel, 1, 1, 1)
        self.conv_y.weight.data = scharr_y.repeat(channel, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, channel)[1]
        self.act = act_layer()

    def forward(self, x):
        # show_feature(x)
        # 应用卷积操作
        edges_x = self.conv_x(x)
        edges_y = self.conv_y(x)
        # 计算边缘和高斯分布强度（可以选择不同的方式进行融合，这里使用平方和开根号）
        scharr_edge = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        scharr_edge = self.act(self.norm(scharr_edge))
        # show_feature(out)

        return scharr_edge


class Gaussian(nn.Module):
    def __init__(self, channel, size, sigma, norm_layer, act_layer):
        super().__init__()
        gaussian = self.gaussian_kernel(size, sigma)
        gaussian = nn.Parameter(data=gaussian, requires_grad=False).clone()
        self.gaussian = nn.Conv2d(channel, channel, kernel_size=size, stride=1, padding=int(size // 2), groups=channel, bias=False)
        self.gaussian.weight.data = gaussian.repeat(channel, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, channel)[1]
        self.act = act_layer()

    def forward(self, x):
        edges_o = self.gaussian(x)
        gaussian = self.act(self.norm(edges_o))
        return gaussian
    
    def gaussian_kernel(self, size: int, sigma: float):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
             for y in range(-size // 2 + 1, size // 2 + 1)
             ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()


class EGA(nn.Module):
    def __init__(self, stage, channel, norm_layer, act_layer):
        super(EGA, self).__init__()
        self.stage = stage
        if stage == 0:
            self.Scharr_edge = Scharr(channel, norm_layer, act_layer)
        else:
            self.gaussian = Gaussian(channel, 5, 1.0, norm_layer, act_layer)
        self.conv_extra = Conv_Extra(channel, norm_layer, act_layer)

    def forward(self, x):
        if self.stage == 0:
            att = self.Scharr_edge(x)
        else:
            att = self.gaussian(x)
        F_a = self.conv_extra(x + att)
        return F_a


class LEG_Module(nn.Module):
    def __init__(self,
                 channel,
                 stage,
                 drop_path,
                 act_layer,
                 norm_layer
                 ):
        super().__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = self.block = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1, dilation=1, bias=False),
            build_norm_layer(norm_layer, channel)[1],
            act_layer())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.norm = build_norm_layer(norm_layer, channel)[1]
        self.EGA = EGA(stage, channel, norm_layer, act_layer)

        mlp_hidden_dim = int(channel * 2)
        mlp_layer: List[nn.Module] = [
            nn.Conv2d(channel, mlp_hidden_dim, 1, bias=False),
            build_norm_layer(norm_layer, mlp_hidden_dim)[1],
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, channel, 1, bias=False)]
        self.mlp = nn.Sequential(*mlp_layer)
        self.norm1 = build_norm_layer(norm_layer, channel)[1]
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        

    def forward(self, x: Tensor) -> Tensor:
        F_a = self.EGA(x)
        F_ega = self.conv2d(x * F_a + x)
        wei = self.avg_pool(F_ega)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x_att = self.norm(x + F_ega * wei)
        out = x + self.norm1(self.drop_path(self.mlp(x_att)))
        return out


class BasicStage(nn.Module):
    def __init__(self,
                 channel,
                 stage,
                 depth,
                 drop_path,
                 norm_layer,
                 act_layer
                 ):
        super().__init__()

        blocks_list = [
            LEG_Module(
                channel=channel,
                stage=stage,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class LoGFilter(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, sigma, norm_layer, act_layer):
        super(LoGFilter, self).__init__()
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_c, out_c, kernel_size=7, stride=1, padding=3)
        """创建高斯-拉普拉斯核"""
        # 初始化二维坐标
        ax = torch.arange(-(kernel_size // 2), (kernel_size // 2) + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax)
        # 计算高斯-拉普拉斯核
        kernel = (xx**2 + yy**2 - 2 * sigma**2) / (2 * math.pi * sigma**4) * torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        # 归一化
        kernel = kernel - kernel.mean()
        kernel = kernel / kernel.sum()
        log_kernel = kernel.unsqueeze(0).unsqueeze(0) # 添加 batch 和 channel 维度
        self.LoG = nn.Conv2d(out_c, out_c, kernel_size=kernel_size, stride=1, padding=int(kernel_size // 2), groups=out_c, bias=False)
        self.LoG.weight.data = log_kernel.repeat(out_c, 1, 1, 1)
        self.act = act_layer()
        self.norm1 = build_norm_layer(norm_layer, out_c)[1]
        self.norm2 = build_norm_layer(norm_layer, out_c)[1]
    
    def forward(self, x):
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        x = self.conv_init(x)  # x = [B, C/4, H, W]
        LoG = self.LoG(x)
        LoG_edge = self.act(self.norm1(LoG))
        x = self.norm2(x + LoG_edge)
        return x
    

class Stem(nn.Module):

    def __init__(self, in_chans, stem_dim, act_layer, norm_layer):
        super().__init__()
        out_c14 = int(stem_dim / 4)  # stem_dim / 2
        out_c12 = int(stem_dim / 2)  # stem_dim / 2
        # original size to 2x downsampling layer
        self.Conv_D = nn.Sequential(
            nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14),
            nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12),
            build_norm_layer(norm_layer, out_c12)[1])
        # 定义LoG滤波器
        self.LoG = LoGFilter(in_chans, out_c14, 7, 1.0, norm_layer, act_layer)
        # gaussian
        self.gaussian = Gaussian(out_c12, 9, 0.5, norm_layer, act_layer)
        self.norm = build_norm_layer(norm_layer, out_c12)[1]
        self.drfd = DRFD(out_c12, norm_layer, act_layer)

    def forward(self, x):
        x = self.LoG(x)
        # original size to 2x downsampling layer
        x = self.Conv_D(x)
        x = self.norm(x + self.gaussian(x))
        x = self.drfd(x)

        return x  # x = [B, C, H/4, W/4]


class LEGNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 stem_dim=32,
                 depths=(1, 4, 4, 2),
                 norm_layer=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU,
                 feature_dim=1000,
                 drop_path_rate=0.1,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.num_features = int(stem_dim * 2 ** (self.num_stages - 1))

        self.Stem = Stem(in_chans=in_chans, stem_dim=stem_dim, act_layer=act_layer, norm_layer=norm_layer)
        
        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(channel=int(stem_dim * 2 ** i_stage),
                               stage=i_stage,
                               depth=depths[i_stage],
                               drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                               norm_layer=norm_layer,
                               act_layer=act_layer
                               )
            stages_list.append(stage)

            # patch merging layer
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    DRFD(channel=int(stem_dim * 2 ** i_stage), norm_layer=norm_layer, act_layer=act_layer)
                )

        self.stages = nn.Sequential(*stages_list)

        self.fork_feat = fork_feat

        if self.fork_feat:
            self.forward = self.forward_det
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    raise NotImplementedError
                else:
                    layer = build_norm_layer(norm_layer, int(stem_dim * 2 ** i_emb))[1]
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            self.forward = self.forward_cls
            # Classifier head
            self.avgpool_pre_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.num_features, feature_dim, 1, bias=False),
                act_layer()
            )
            self.head = nn.Linear(feature_dim, num_classes) \
                if num_classes > 0 else nn.Identity()

        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # init for mmdetection by loading imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def forward_cls(self, x):
        # output only the features of last layer for image classification
        x = self.Stem(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)  # B C 1 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

    def forward_det(self, x: Tensor) -> Tensor:
        # output the features of four stages for dense prediction
        x = self.Stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        # return outs
        return tuple(outs)
