# Copyright (c) Open-MMLab. All rights reserved.
import warnings
from abc import ABCMeta
import torch.nn as nn


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base module for all modules in openmmlab."""

    def __init__(self, init_cfg=None):
        """初始化BaseModule，继承自`torch.nn.Module`

            参数:
                init_cfg (dict, 可选): 初始化配置字典。
        """
        # 注意：init_cfg 可以定义在不同的级别，但低级别的 init_cfg 优先级更高。

        super(BaseModule, self).__init__()
        # 将 init_cfg 的默认值定义在 init_weight () 函数之外，而不是在函数内部硬编码。
        self._is_init = False
        self.init_cfg = init_cfg

        '''
        为了向下兼容派生类
        如果 pretrained 不为 None：
            发出警告： pretrained 是一个已弃用的关键字，请考虑使用 init_cfg。
            将 self.init_cfg 设置为  dict(type='Pretrained', checkpoint=pretrained)
        '''

    @property
    def is_init(self):
        return self._is_init

    def init_weights(self):
        """初始化模型权重。"""
        from utils import initialize
        if not self._is_init:
            if self.init_cfg:
                initialize(self, self.init_cfg)
                if isinstance(self.init_cfg, (dict)):
                    # Avoid the parameters of the pre-training model being overwritten by the init_weights of the children.
                    if self.init_cfg['type'] == 'Pretrained':
                        return

            for m in self.children():
                if hasattr(m, 'init_weights'):
                    m.init_weights()
            self._is_init = True
        else:
            warnings.warn(f'init_weights of {self.__class__.__name__} has been called more than once.')

    def __repr__(self):
        """返回模型的字符串表示。"""
        s = super().__repr__()
        if self.init_cfg:
            s += f'\ninit_cfg={self.init_cfg}'
        return s


class Sequential(BaseModule, nn.Sequential):
    """openmmlab 中的顺序模块。

    Args：
        init_cfg （dict， 可选）： 初始化配置 dict.
    """

    def __init__(self, *args, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.Sequential.__init__(self, *args)


class ModuleList(BaseModule, nn.ModuleList):
    """openmmlab 中的 ModuleList。

    Args:
        modules （iterable， optional）：要添加的模块的可迭代对象。
        init_cfg （dict， optional）： 初始化配置 dict。
    """

    def __init__(self, modules=None, init_cfg=None):
        BaseModule.__init__(self, init_cfg)
        nn.ModuleList.__init__(self, modules)
