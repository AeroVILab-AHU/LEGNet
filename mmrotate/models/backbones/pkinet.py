import math
from typing import Optional, Union, Sequence

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, constant_init
from mmengine.model.weight_init import trunc_normal_init, normal_init
from mmengine.logging import MMLogger

from mmrotate.models.builder import ROTATED_BACKBONES
from mmrotate.models.utils import autopad, make_divisible, BHWC2BCHW, BCHW2BHWC


class GSiLU(BaseModule):
    """Global Sigmoid-Gated Linear Unit, reproduced from paper <SIMPLE CNN FOR VISION>"""
    def __init__(self):
        super().__init__()
        self.adpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.adpool(x))


class CAA(BaseModule):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1,
                                 (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1,
                                 (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor


class ConvFFN(BaseModule):
    """Multi-layer perceptron implemented with ConvModule"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            hidden_channels_scale: float = 4.0,
            hidden_kernel_size: int = 3,
            dropout_rate: float = 0.,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = int(in_channels * hidden_channels_scale)

        self.ffn_layers = nn.Sequential(
            BCHW2BHWC(),
            nn.LayerNorm(in_channels),
            BHWC2BCHW(),
            ConvModule(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(hidden_channels, hidden_channels, kernel_size=hidden_kernel_size, stride=1,
                       padding=hidden_kernel_size // 2, groups=hidden_channels,
                       norm_cfg=norm_cfg, act_cfg=None),
            GSiLU(),
            nn.Dropout(dropout_rate),
            ConvModule(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Dropout(dropout_rate),
        )
        self.add_identity = add_identity

    def forward(self, x):
        x = x + self.ffn_layers(x) if self.add_identity else self.ffn_layers(x)
        return x


class Stem(BaseModule):
    """Stem layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion: float = 1.0,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.down_conv = ConvModule(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv1 = ConvModule(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.conv2(self.conv1(self.down_conv(x)))


class DownSamplingLayer(BaseModule):
    """Down sampling layer"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or (in_channels * 2)

        self.down_conv = ConvModule(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        return self.down_conv(x)


class InceptionBottleneck(BaseModule):
    """Bottleneck with Inception module"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 1.0,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.pre_conv = ConvModule(in_channels, hidden_channels, 1, 1, 0, 1,
                                   norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.dw_conv = ConvModule(hidden_channels, hidden_channels, kernel_sizes[0], 1,
                                  autopad(kernel_sizes[0], None, dilations[0]), dilations[0],
                                  groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv1 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[1], 1,
                                   autopad(kernel_sizes[1], None, dilations[1]), dilations[1],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv2 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[2], 1,
                                   autopad(kernel_sizes[2], None, dilations[2]), dilations[2],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv3 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[3], 1,
                                   autopad(kernel_sizes[3], None, dilations[3]), dilations[3],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.dw_conv4 = ConvModule(hidden_channels, hidden_channels, kernel_sizes[4], 1,
                                   autopad(kernel_sizes[4], None, dilations[4]), dilations[4],
                                   groups=hidden_channels, norm_cfg=None, act_cfg=None)
        self.pw_conv = ConvModule(hidden_channels, hidden_channels, 1, 1, 0, 1,
                                  norm_cfg=norm_cfg, act_cfg=act_cfg)

        if with_caa:
            self.caa_factor = CAA(hidden_channels, caa_kernel_size, caa_kernel_size, None, None)
        else:
            self.caa_factor = None

        self.add_identity = add_identity and in_channels == out_channels

        self.post_conv = ConvModule(hidden_channels, out_channels, 1, 1, 0, 1,
                                    norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.pre_conv(x)

        y = x  # if there is an inplace operation of x, use y = x.clone() instead of y = x
        x = self.dw_conv(x)
        x = x + self.dw_conv1(x) + self.dw_conv2(x) + self.dw_conv3(x) + self.dw_conv4(x)
        x = self.pw_conv(x)
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x


class PKIBlock(BaseModule):
    """Poly Kernel Inception Block"""
    def __init__(
            self,
            in_channels: int,
            out_channels: Optional[int] = None,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            expansion: float = 1.0,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale: Optional[float] = 1.0,
            add_identity: bool = True,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        out_channels = out_channels or in_channels
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        if norm_cfg is not None:
            self.norm1 = build_norm_layer(norm_cfg, in_channels)[1]
            self.norm2 = build_norm_layer(norm_cfg, hidden_channels)[1]
        else:
            self.norm1 = nn.BatchNorm2d(in_channels)
            self.norm2 = nn.BatchNorm2d(hidden_channels)

        self.block = InceptionBottleneck(in_channels, hidden_channels, kernel_sizes, dilations,
                                         expansion=1.0, add_identity=True,
                                         with_caa=with_caa, caa_kernel_size=caa_kernel_size,
                                         norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.ffn = ConvFFN(hidden_channels, out_channels, ffn_scale, ffn_kernel_size, dropout_rate, add_identity=False,
                           norm_cfg=None, act_cfg=None)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.layer_scale = layer_scale
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_channels), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(out_channels), requires_grad=True)
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x):
        if self.layer_scale:
            if self.add_identity:
                x = x + self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = x + self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.gamma1.unsqueeze(-1).unsqueeze(-1) * self.block(self.norm1(x)))
                x = self.drop_path(self.gamma2.unsqueeze(-1).unsqueeze(-1) * self.ffn(self.norm2(x)))
        else:
            if self.add_identity:
                x = x + self.drop_path(self.block(self.norm1(x)))
                x = x + self.drop_path(self.ffn(self.norm2(x)))
            else:
                x = self.drop_path(self.block(self.norm1(x)))
                x = self.drop_path(self.ffn(self.norm2(x)))
        return x


class PKIStage(BaseModule):
    """Poly Kernel Inception Stage"""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            kernel_sizes: Sequence[int] = (3, 5, 7, 9, 11),
            dilations: Sequence[int] = (1, 1, 1, 1, 1),
            expansion: float = 0.5,
            ffn_scale: float = 4.0,
            ffn_kernel_size: int = 3,
            dropout_rate: float = 0.,
            drop_path_rate: Union[float, list] = 0.,
            layer_scale: Optional[float] = 1.0,
            shortcut_with_ffn: bool = True,
            shortcut_ffn_scale: float = 4.0,
            shortcut_ffn_kernel_size: int = 5,
            add_identity: bool = True,
            with_caa: bool = True,
            caa_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = None,
    ):
        super().__init__(init_cfg)
        hidden_channels = make_divisible(int(out_channels * expansion), 8)

        self.downsample = DownSamplingLayer(in_channels, out_channels, norm_cfg, act_cfg)

        self.conv1 = ConvModule(out_channels, 2 * hidden_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(2 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(out_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.ffn = ConvFFN(hidden_channels, hidden_channels, shortcut_ffn_scale, shortcut_ffn_kernel_size, 0.,
                           add_identity=True, norm_cfg=None, act_cfg=None) if shortcut_with_ffn else None

        self.blocks = nn.ModuleList([
            PKIBlock(hidden_channels, hidden_channels, kernel_sizes, dilations, with_caa,
                     caa_kernel_size+2*i, 1.0, ffn_scale, ffn_kernel_size, dropout_rate,
                     drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                     layer_scale, add_identity, norm_cfg, act_cfg) for i in range(num_blocks)
        ])

    def forward(self, x):
        x = self.downsample(x)

        x, y = list(self.conv1(x).chunk(2, 1))
        if self.ffn is not None:
            x = self.ffn(x)

        z = [x]
        t = torch.zeros(y.shape, device=y.device)
        for block in self.blocks:
            t = t + block(y)
        z.append(t)
        z = torch.cat(z, dim=1)
        z = self.conv2(z)
        z = self.conv3(z)

        return z


@ROTATED_BACKBONES.register_module()
class PKINet(BaseModule):
    """Poly Kernel Inception Network"""
    arch_settings = {
        # from left to right: (indices)
        # in_channels(0), out_channels(1), num_blocks(2), kernel_sizes(3), dilations(4), expansion(5),
        # ffn_scale(6), ffn_kernel_size(7), dropout_rate(8), layer_scale(9), shortcut_with_ffn(10),
        # shortcut_ffn_scale(11), shortcut_ffn_kernel_size(12), add_identity(13), with_caa(14), caa_kernel_size(15)
        'T': [[16, 32, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 5, True, True, 11],
              [32, 64, 14, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 7, True, True, 11],
              [64, 128, 22, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 9, True, True, 11],
              [128, 256, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 11, True, True, 11]],

        'S': [[32, 64, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 5, True, True, 11],
              [64, 128, 12, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 7, True, True, 11],
              [128, 256, 20, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 9, True, True, 11],
              [256, 512, 4, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 11, True, True, 11]],

        'B': [[40, 80, 6, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 5, True, True, 11],
              [80, 160, 16, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 8.0, 7, True, True, 11],
              [160, 320, 24, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 9, True, True, 11],
              [320, 640, 6, (3, 5, 7, 9, 11), (1, 1, 1, 1, 1), 0.5, 4.0, 3, 0.1, 1.0, True, 4.0, 11, True, True, 11]],
    }

    def __init__(
            self,
            arch: str = 'S',
            out_indices: Sequence[int] = (2, 3, 4),
            drop_path_rate: float = 0.1,
            frozen_stages: int = -1,
            norm_eval: bool = False,
            arch_setting: Optional[Sequence[list]] = None,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU'),
            init_cfg: Optional[dict] = dict(type='Kaiming',
                                            layer='Conv2d',
                                            a=math.sqrt(5),
                                            distribution='uniform',
                                            mode='fan_in',
                                            nonlinearity='leaky_relu'),
    ):
        super().__init__(init_cfg=init_cfg)
        arch_setting = arch_setting or self.arch_settings[arch]

        assert set(out_indices).issubset(i for i in range(len(arch_setting) + 1))
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError(f'frozen_stages must be in range(-1, len(arch_setting) + 1). But received {frozen_stages}')

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.stages = nn.ModuleList()

        self.stem = Stem(3, arch_setting[0][0], expansion=1.0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.stages.append(self.stem)

        depths = [x[2] for x in arch_setting]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i, (in_channels, out_channels, num_blocks, kernel_sizes, dilations, expansion, ffn_scale, ffn_kernel_size,
                dropout_rate, layer_scale, shortcut_with_ffn, shortcut_ffn_scale, shortcut_ffn_kernel_size,
                add_identity, with_caa, caa_kernel_size) in enumerate(arch_setting):
            stage = PKIStage(in_channels, out_channels, num_blocks, kernel_sizes, dilations, expansion,
                             ffn_scale, ffn_kernel_size, dropout_rate, dpr[sum(depths[:i]):sum(depths[:i + 1])],
                             layer_scale, shortcut_with_ffn, shortcut_ffn_scale, shortcut_ffn_kernel_size,
                             add_identity, with_caa, caa_kernel_size, norm_cfg, act_cfg)
            self.stages.append(stage)

    def forward(self, x):
        outs = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warning(f'No pre-trained weights for {self.__class__.__name__}, training start from scratch.')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super().init_weights()

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = self.stages[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
