import math
import torch
from torch import nn
from torch.nn import functional as F
from model.tools.ConvModule import ConvModule
from utils import constant_init


class _SelfAttentionBlock(nn.Module):
    """General self-attention block/non-local block.

    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out,
                 conv_cfg, norm_cfg, act_cfg):
        super(SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context


class SelfAttentionBlock(_SelfAttentionBlock):
    """Self-Attention Module.

    Args:
        in_channels (int): Input channels of key/query feature.
        channels (int): Output channels of key/query transform.
        conv_cfg (dict | None): Config of conv layers.
        norm_cfg (dict | None): Config of norm layers.
        act_cfg (dict | None): Config of activation layers.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 conv_cfg,
                 norm_cfg,
                 act_cfg,
                 key_query_num_convs=2):
        super(SelfAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=key_query_num_convs,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=True,
            with_out=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.output_project = self.build_project(
            in_channels,
            in_channels,
            num_convs=1,
            use_conv_module=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        context = super(SelfAttentionBlock, self).forward(x, x)
        return self.output_project(context)


class ISALayer(nn.Module):

    def __init__(self,
                 in_channels,
                 channels,
                 isa_channels,
                 down_factor=(8, 8),
                 key_query_num_convs=2,
                 in_conv_kernel_size=1,
                 out_cat_and_conv=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super(ISALayer, self).__init__()
        self.down_factor = down_factor
        self.out_cat_and_conv = out_cat_and_conv

        if in_conv_kernel_size is not None:
            self.in_conv = ConvModule(
                in_channels,
                channels,
                kernel_size=in_conv_kernel_size,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.in_conv = None
        self.global_relation = SelfAttentionBlock(
            channels,
            isa_channels,
            key_query_num_convs=key_query_num_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.local_relation = SelfAttentionBlock(
            channels,
            isa_channels,
            key_query_num_convs=key_query_num_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if out_cat_and_conv:
            self.out_conv = ConvModule(
                channels * 2,
                channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        if self.in_conv is not None:
            x = self.in_conv(x)
        if self.out_cat_and_conv:
            residual = x
        n, c, h, w = x.size()
        loc_h, loc_w = self.down_factor  # size of local group in H- and W-axes
        glb_h, glb_w = math.ceil(h / loc_h), math.ceil(w / loc_w)
        pad_h, pad_w = glb_h * loc_h - h, glb_w * loc_w - w
        if pad_h > 0 or pad_w > 0:  # pad if the size is not divisible
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                       pad_h - pad_h // 2)
            x = F.pad(x, padding)

        # global relation
        x = x.view(n, c, glb_h, loc_h, glb_w, loc_w)
        # do permutation to gather global group
        x = x.permute(0, 3, 5, 1, 2, 4)  # (n, loc_h, loc_w, c, glb_h, glb_w)
        x = x.reshape(-1, c, glb_h, glb_w)
        # apply attention within each global group
        x = self.global_relation(x)  # (n * loc_h * loc_w, c, glb_h, glb_w)

        # local relation
        x = x.view(n, loc_h, loc_w, c, glb_h, glb_w)
        # do permutation to gather local group
        x = x.permute(0, 4, 5, 3, 1, 2)  # (n, glb_h, glb_w, c, loc_h, loc_w)
        x = x.reshape(-1, c, loc_h, loc_w)
        # apply attention within each local group
        x = self.local_relation(x)  # (n * glb_h * glb_w, c, loc_h, loc_w)

        # permute each pixel back to its original position
        x = x.view(n, glb_h, glb_w, c, loc_h, loc_w)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (n, c, glb_h, loc_h, glb_w, loc_w)
        x = x.reshape(n, c, glb_h * loc_h, glb_w * loc_w)
        if pad_h > 0 or pad_w > 0:  # remove padding
            x = x[:, :, pad_h // 2:pad_h // 2 + h, pad_w // 2:pad_w // 2 + w]

        if self.out_cat_and_conv:
            x = self.out_conv(torch.cat([x, residual], dim=1))

        return x