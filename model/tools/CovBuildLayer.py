import inspect

import torch
import torch.nn as nn


activation_module = [
        'ReLU', 'LeakyReLU', 'PReLU', 'RReLU', 'ReLU6', 'ELU',
        'Sigmoid', 'Tanh', 'GELU'
]

def build_activation_layer(cfg, *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type='ReLU')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()
    act_layer = cfg_.pop('type')
    if act_layer not in activation_module:
        raise KeyError(f'Unrecognized norm type {activation_module}')
    # 检查激活层类型是否在 nn 模块中
    if act_layer in nn.__dict__:
        act_layer = getattr(nn,act_layer)(*args, **kwargs, **cfg_)
    else:
        raise KeyError(f'Activation type {act_layer} is not found in torch.nn')

    # 构建并返回激活层
    return act_layer


convlution_module = [
    'Conv1d', 'Conv2d', 'Conv3d', 'Conv',
    'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d'
]

def build_conv_layer(cfg, *args, **kwargs):
    # 检查参数表合法性
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()
    # 处理参数表内容
    conv_layer = cfg_.pop('type')
    if conv_layer not in convlution_module:
        raise KeyError(f'Unrecognized norm type {convlution_module}')
    # 检查激活层类型是否在 nn 模块中
    if conv_layer in nn.__dict__:
        conv_layer = getattr(nn,conv_layer)(*args, **kwargs, **cfg_)
    else:
        raise KeyError(f'Conv type {conv_layer} is not found in torch.nn')
    # conv_layer = conv_layer()
    return conv_layer


normalization_module = [
    'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'BatchNorm',
    'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d', 'InstanceNorm',
    'LayerNorm', 'GroupNorm', 'SyncBatchNorm', 'LocalResponseNorm',
    'BN', 'BN1d', 'BN2d', 'BN3d', 'IN', 'IN1d', 'IN2d', 'IN3d', 'SyncBN', 'GN', 'LN', 'LRN'
]

def get_norm(name):
    return {
        'BN': nn.BatchNorm2d,
        'BN1d': nn.BatchNorm1d,
        'BN2d': nn.BatchNorm2d,
        'BN3d': nn.BatchNorm3d,

        'IN': nn.InstanceNorm2d,
        'IN1d': nn.InstanceNorm1d,
        'IN2d': nn.InstanceNorm2d,
        'IN3d': nn.InstanceNorm3d,

        'SyncBN': nn.SyncBatchNorm,
        'GN': nn.GroupNorm,
        'LN': nn.LayerNorm,
        'LRN': nn.LocalResponseNorm
    }[name]

def build_norm_layer(cfg, num_features, postfix=''):
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    if layer_type not in normalization_module:
        raise KeyError(f'Unrecognized norm type {normalization_module}')
    norm_layer = get_norm(layer_type)

    abbr = infer_abbr(norm_layer)
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


TORCH_VERSION = torch.__version__

def _get_norm():
    from torch.nn.modules.instancenorm import _InstanceNorm
    from torch.nn.modules.batchnorm import _BatchNorm
    SyncBatchNorm_ = torch.nn.SyncBatchNorm
    return _BatchNorm, _InstanceNorm, SyncBatchNorm_

_BatchNorm, _InstanceNorm, SyncBatchNorm_ = _get_norm()

def infer_abbr(class_type):
    """从类名中推断出规范化层的缩写。 """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    if issubclass(class_type, _InstanceNorm):  # IN is a subclass of BN
        return 'in'
    elif issubclass(class_type, _BatchNorm):
        return 'bn'
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'

padding_module = ['zero', 'reflect', 'replicate']

def build_padding_layer(cfg, *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type='zero')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if padding_type not in padding_module:
        raise KeyError(f'Unrecognized padding type {padding_module}')
    if padding_type in nn.__dict__:
        padding_layer = getattr(nn, padding_type)()
    else:
        raise KeyError(f'Padding type {padding_type} is not found in torch.nn')
    return padding_layer(*args, **kwargs, **cfg_)