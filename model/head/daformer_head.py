# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
import copy
from copy import deepcopy

import torch
import torch.nn as nn

from model.base_module import BaseModule
from model.head.decode_head import BaseDecodeHead
from model.tools.ConvModule import ConvModule, DepthwiseSeparableConvModule
from model.tools.aspp_module import ASPPWrapper
from model.tools.crossEntropyLoss import CrossEntropyLoss
from model.tools.isa_Layer import ISALayer
from utils.wrappers import wrap_resize


class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


def build_layer(in_channels, out_channels, type, **kwargs):
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(in_channels=in_channels, out_channels=out_channels, padding=kwargs['kernel_size'] // 2, **kwargs)
    elif type == 'conv':
        return ConvModule(in_channels=in_channels, out_channels=out_channels, padding=kwargs['kernel_size'] // 2, **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            ASPPWrapper(in_channels=out_channels, channels=out_channels, **kwargs))
    elif type == 'isa':
        return ISALayer(in_channels=in_channels, channels=out_channels, **kwargs)
    else:
        raise NotImplementedError(type)


class DAFormerHead(BaseDecodeHead):

    def __init__(self, **kwargs):
        super(DAFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        # DAFormerHead中的构建
        assert not self.align_corners
        decoder_params = kwargs['decoder_params']
        embed_dims = decoder_params['embed_dims']
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)
        embed_cfg = decoder_params['embed_cfg']
        embed_neck_cfg = decoder_params['embed_neck_cfg']
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg

        fusion_cfg = decoder_params['fusion_cfg']
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels, embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(in_channels, embed_dim, **embed_neck_cfg)
            else:
                self.embed_layers[str(i)] = build_layer(in_channels, embed_dim, **embed_cfg)
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(sum(embed_dims), self.channels, **fusion_cfg)      # ASPPWrapper

    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     print(f'{f.shape}', 'mmseg')

        os_size = x[0].size()[2:]
        _c = {}
        for i in self.in_index:
            # print(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:
                _c[i] = _c[i].permute(0, 2, 1).contiguous().reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # print(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # print(f'resize {i}', 'mmseg')
                _c[i] = wrap_resize(
                    _c[i],
                    size=os_size,
                    mode='bilinear',
                    align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        x = self.cls_seg(x)
        return x

    # --------------------------------------decode_head中的实现--------------------------------------

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    # def _init_inputs(self, in_channels, in_index, input_transform):
    #     """Check and initialize input transforms.        """
    #
    #     if input_transform is not None:
    #         assert input_transform in ['resize_concat', 'multiple_select']
    #     self.input_transform = input_transform
    #     self.in_index = in_index
    #     if input_transform is not None:
    #         assert isinstance(in_channels, (list, tuple))
    #         assert isinstance(in_index, (list, tuple))
    #         assert len(in_channels) == len(in_index)
    #         if input_transform == 'resize_concat':
    #             self.in_channels = sum(in_channels)
    #         else:
    #             self.in_channels = in_channels
    #     else:
    #         assert isinstance(in_channels, int)
    #         assert isinstance(in_index, int)
    #         self.in_channels = in_channels

    # def init_weights(self):
    #     """Initialize the weights."""
    #     import warnings
    #     from collections import defaultdict
    #     from utils.weight_init import initialize, update_init_info
    #     is_top_level_module = False
    #     if not hasattr(self, '_params_init_info'):
    #         self._params_init_info = defaultdict(dict)
    #         is_top_level_module = True
    #         for name, param in self.named_parameters():
    #             self._params_init_info[param]['init_info'] = f'The value is the same before and after calling `init_weights` of {self.__class__.__name__} '
    #             self._params_init_info[param]['tmp_mean_value'] = param.data.mean()
    #         for sub_module in self.modules():
    #             sub_module._params_init_info = self._params_init_info
    #     module_name = self.__class__.__name__
    #     if not self.is_init:
    #         if self.init_cfg:
    #             print(f'initialize {module_name} with init_cfg {self.init_cfg}')
    #             initialize(self, self.init_cfg)
    #             if isinstance(self.init_cfg, dict):
    #                 if self.init_cfg['type'] == 'Pretrained':
    #                     return
    #         for m in self.children():
    #             if hasattr(m, 'init_weights'):
    #                 m.init_weights()
    #                 update_init_info(m, init_info=f'Initialized by user-defined `init_weights` in {m.__class__.__name__} ')
    #         self.is_init = True
    #     else:
    #         warnings.warn(f'init_weights of {self.__class__.__name__} has been called more than once.')
    #     if is_top_level_module:
    #         for sub_module in self.modules():
    #             del sub_module._params_init_info

