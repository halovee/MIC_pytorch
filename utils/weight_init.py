# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import warnings
from model.runner.hooks import load_checkpoint, _load_checkpoint_with_prefix
from utils import get_dist_info


def update_init_info(module, init_info):
    """Update the `_params_init_info` in the module if the value of parameters
    are changed.

    Args:
        module (obj:`nn.Module`): The module of PyTorch with a user-defined
            attribute `_params_init_info` which records the initialization
            information.
        init_info (str): The string that describes the initialization.
    """
    assert hasattr(module, '_params_init_info'), f'Can not find `_params_init_info` in {module}'
    for name, param in module.named_parameters():

        assert param in module._params_init_info, (
            f'Find a new :obj:`Parameter` named `{name}` during executing the `init_weights` of `{module.__class__.__name__}`. '
            f'Please do not add or replace parameters during executing the `init_weights`. ')

        # The parameter has been changed during executing the `init_weights` of module
        mean_value = param.data.mean()
        if module._params_init_info[param]['tmp_mean_value'] != mean_value:
            module._params_init_info[param]['init_info'] = init_info
            module._params_init_info[param]['tmp_mean_value'] = mean_value


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore


def uniform_init(module, a=0, b=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def caffe2_xavier_init(module, bias=0):
    # `XavierFill` in Caffe2 corresponds to `kaiming_uniform_` in PyTorch
    # Acknowledgment to FAIR's internal code
    kaiming_init(
        module,
        a=1,
        mode='fan_in',
        nonlinearity='leaky_relu',
        bias=bias,
        distribution='uniform')


def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def _get_bases_name(m):
    return [b.__name__ for b in m.__class__.__bases__]


class BaseInit(object):

    def __init__(self, *, bias=0, bias_prob=None, layer=None):
        self.wholemodule = False
        if not isinstance(bias, (int, float)):
            raise TypeError(f'bias must be a number, but got a {type(bias)}')

        if bias_prob is not None:
            if not isinstance(bias_prob, float):
                raise TypeError(f'bias_prob type must be float, but got {type(bias_prob)}')

        if layer is not None:
            if not isinstance(layer, (str, list)):
                raise TypeError(f'layer must be a str or a list of str, but got a {type(layer)}')
        else:
            layer = []

        if bias_prob is not None:
            self.bias = bias_init_with_prob(bias_prob)
        else:
            self.bias = bias
        self.layer = [layer] if isinstance(layer, str) else layer

    def _get_init_info(self):
        info = f'{self.__class__.__name__}, bias={self.bias}'
        return info


class ConstantInit(BaseInit):
    """Initialize module parameters with constant values.

    Args:
        val (int | float): the value to fill the weights in the module with
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, val, **kwargs):
        super().__init__(**kwargs)
        self.val = val

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                constant_init(m, self.val, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    constant_init(m, self.val, self.bias)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: val={self.val}, bias={self.bias}'
        return info


class XavierInit(BaseInit):
    r"""Initialize module parameters with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks - Glorot, X. & Bengio, Y. (2010).
    <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_

    Args:
        gain (int | float): an optional scaling factor. Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        distribution (str): distribution either be ``'normal'``
            or ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, gain=1, distribution='normal', **kwargs):
        super().__init__(**kwargs)
        self.gain = gain
        self.distribution = distribution

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                xavier_init(m, self.gain, self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    xavier_init(m, self.gain, self.bias, self.distribution)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: gain={self.gain}, ' \
               f'distribution={self.distribution}, bias={self.bias}'
        return info



class NormalInit(BaseInit):
    r"""Initialize module parameters with the values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.

    Args:
        mean (int | float):the mean of the normal distribution. Defaults to 0.
        std (int | float): the standard deviation of the normal distribution.
            Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.

    """

    def __init__(self, mean=0, std=1, **kwargs):
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                normal_init(m, self.mean, self.std, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    normal_init(m, self.mean, self.std, self.bias)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: mean={self.mean}, std={self.std}, bias={self.bias}'
        return info



class TruncNormalInit(BaseInit):
    r"""Initialize module parameters with the values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)` with values
    outside :math:`[a, b]`.

    Args:
        mean (float): the mean of the normal distribution. Defaults to 0.
        std (float):  the standard deviation of the normal distribution.
            Defaults to 1.
        a (float): The minimum cutoff value.
        b ( float): The maximum cutoff value.
        bias (float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.

    """

    def __init__(self,
                 mean: float = 0,
                 std: float = 1,
                 a: float = -2,
                 b: float = 2,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

    def __call__(self, module: nn.Module) -> None:

        def init(m):
            if self.wholemodule:
                trunc_normal_init(m, self.mean, self.std, self.a, self.b,
                                  self.bias)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    trunc_normal_init(m, self.mean, self.std, self.a, self.b,
                                      self.bias)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: a={self.a}, b={self.b},' \
               f' mean={self.mean}, std={self.std}, bias={self.bias}'
        return info



class UniformInit(BaseInit):
    r"""Initialize module parameters with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.

    Args:
        a (int | float): the lower bound of the uniform distribution.
            Defaults to 0.
        b (int | float): the upper bound of the uniform distribution.
            Defaults to 1.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self, a=0, b=1, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.b = b

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                uniform_init(m, self.a, self.b, self.bias)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    uniform_init(m, self.a, self.b, self.bias)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: a={self.a},' \
               f' b={self.b}, bias={self.bias}'
        return info



class KaimingInit(BaseInit):
    r"""Initialize module parameters with the values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification - He, K. et al. (2015).
    <https://www.cv-foundation.org/openaccess/content_iccv_2015/
    papers/He_Delving_Deep_into_ICCV_2015_paper.pdf>`_

    Args:
        a (int | float): the negative slope of the rectifier used after this
            layer (only used with ``'leaky_relu'``). Defaults to 0.
        mode (str):  either ``'fan_in'`` or ``'fan_out'``. Choosing
            ``'fan_in'`` preserves the magnitude of the variance of the weights
            in the forward pass. Choosing ``'fan_out'`` preserves the
            magnitudes in the backwards pass. Defaults to ``'fan_out'``.
        nonlinearity (str): the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` .
            Defaults to 'relu'.
        bias (int | float): the value to fill the bias. Defaults to 0.
        bias_prob (float, optional): the probability for bias initialization.
            Defaults to None.
        distribution (str): distribution either be ``'normal'`` or
            ``'uniform'``. Defaults to ``'normal'``.
        layer (str | list[str], optional): the layer will be initialized.
            Defaults to None.
    """

    def __init__(self,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 distribution='normal',
                 **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.distribution = distribution

    def __call__(self, module):

        def init(m):
            if self.wholemodule:
                kaiming_init(m, self.a, self.mode, self.nonlinearity,
                             self.bias, self.distribution)
            else:
                layername = m.__class__.__name__
                basesname = _get_bases_name(m)
                if len(set(self.layer) & set([layername] + basesname)):
                    kaiming_init(m, self.a, self.mode, self.nonlinearity,
                                 self.bias, self.distribution)

        module.apply(init)
        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: a={self.a}, mode={self.mode}, ' \
               f'nonlinearity={self.nonlinearity}, ' \
               f'distribution ={self.distribution}, bias={self.bias}'
        return info



class Caffe2XavierInit(KaimingInit):
    # Caffe2 中的“XavierFill”对应于 PyTorch 中对 FAIR 内部代码的“kaiming_uniform_”的确认
    def __init__(self, **kwargs):
        super().__init__(
            a=1,
            mode='fan_in',
            nonlinearity='leaky_relu',
            distribution='uniform',
            **kwargs)

    def __call__(self, module):
        super().__call__(module)



class PretrainedInit(object):
    """Initialize module by loading a pretrained model.

    Args:
        checkpoint (str): the checkpoint file of the pretrained model should be load.
        prefix (str, optional): the prefix of a sub-module in the pretrained model.
            it is for loading a part of the pretrained model to initialize.
            For example, if we would like to only load the backbone of a detector model, we can set ``prefix='backbone.'``.
            Defaults to None.
        map_location (str): map tensors into proper locations.
    """

    def __init__(self, checkpoint, prefix=None, map_location=None):
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.map_location = map_location

    def __call__(self, module):

        if self.prefix is None:
            print(f'load model from: {self.checkpoint}')
            load_checkpoint(module, self.checkpoint, map_location=self.map_location, strict=False)
        else:
            print(f'load {self.prefix} in model from: {self.checkpoint}')
            state_dict = _load_checkpoint_with_prefix(self.prefix, self.checkpoint, map_location=self.map_location)
            load_state_dict(module, state_dict, strict=False)

        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: load from {self.checkpoint}'
        return info

def _initialize_override(module, override, cfg):
    if not isinstance(override, (dict, list)):
        raise TypeError(f'override must be a dict or a list of dict, but got {type(override)}')

    override = [override] if isinstance(override, dict) else override
    for override_ in override:
        cp_override = copy.deepcopy(override_)
        name = cp_override.pop('name', None)
        if name is None:
            raise ValueError('`override` must contain the key "name",but got {cp_override}')
        if not cp_override:
            cp_override.update(cfg)
        elif 'type' not in cp_override.keys():
            raise ValueError(f'`override` need "type" key, but got {cp_override}')
        if hasattr(module, name):
            _initialize(getattr(module, name), cp_override, wholemodule=True)
        else:
            raise RuntimeError(f'module did not have attribute {name}, but init_cfg is {cp_override}.')

def get_initClass(obj_type):
    module_dict = {
        'Constant': ConstantInit,
        'Xavier': XavierInit,
        'Normal': NormalInit,
        'TruncNormal': TruncNormalInit,
        'Uniform': UniformInit,
        'Kaiming': KaimingInit,
        'Pretrained': PretrainedInit,
        'Caffe2Xavier': Caffe2XavierInit,
    }
    return module_dict[obj_type]

def _initialize(module, cfg, wholemodule=False):
    INITIALIZERS=['Constant', 'Xavier', 'Normal', 'TruncNormal', 'Uniform', 'Kaiming', 'Pretrained', 'Caffe2Xavier']
    args = cfg.copy()
    obj_type = args.pop('type')
    if obj_type not in INITIALIZERS:
        raise ValueError(f'Unsupported init type: {obj_type}')
    else:
        obj_cls = get_initClass(obj_type)
        func = obj_cls(**args)
        # wholeModule 标志用于 override 模式，override 中没有层键，初始化器将为整个模块提供 init 值，其名称为 override。
        func.wholemodule = wholemodule
        func(module)

def initialize(module, init_cfg):
    """Initialize a module.    """
    if not isinstance(init_cfg, (dict, list)):
        raise TypeError(f'init_cfg must be a dict or a list of dict, but got {type(init_cfg)}')

    if isinstance(init_cfg, dict):
        init_cfg = [init_cfg]

    for cfg in init_cfg:
        cp_cfg = copy.deepcopy(cfg)
        override = cp_cfg.pop('override', None)
        _initialize(module, cp_cfg)

        if override is not None:
            cp_cfg.pop('layer', None)
            _initialize_override(module, override, cp_cfg)
        else:
            pass

def _no_grad_trunc_normal_(tensor: Tensor, mean: float, std: float, a: float, b: float) -> Tensor:

    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor: Tensor,
                  mean: float = 0.,
                  std: float = 1.,
                  a: float = -2.,
                  b: float = 2.) -> Tensor:
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def load_state_dict(module, state_dict, strict=False):
    """Load state_dict to a module.  """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # 递归检查 Parallel Module，以防模型具有复杂的结构，例如 nn.模块（nn.模块 （DDP））
        # if is_module_wrapper(module):
        #     module = module.module
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, all_missing_keys, unexpected_keys, err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None  # break load->load reference cycle

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [key for key in all_missing_keys if 'num_batches_tracked' not in key]
    if unexpected_keys:
        err_msg.append(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            print(err_msg)
