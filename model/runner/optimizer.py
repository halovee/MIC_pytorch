import copy
import inspect
import torch
from model.runner.optimizer_constructor import DefaultOptimizerConstructor

def build_optimizer(model, cfg):
    optimizer_cfg = copy.deepcopy(cfg)
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optimizer_cfg.pop('constructor', 'DefaultOptimizerConstructor')
    optim_constructor = DefaultOptimizerConstructor(
        optimizer_cfg=optimizer_cfg,
        paramwise_cfg=paramwise_cfg)
    optimizer = optim_constructor(model)   # 在这里调用了DefaultOptimizerConstructor的__call__方法
    return optimizer

def register_torch_optimizers():
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim, torch.optim.Optimizer):
            torch_optimizers.append(module_name)
    return torch_optimizers

TORCH_OPTIMIZERS = register_torch_optimizers()
#　['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']

def build_optimizer_from_cfg(cfg):
    args = cfg.copy()
    obj_type = args.pop('type')     # 'AdamW'
    assert 'AdamW' in TORCH_OPTIMIZERS, f'{obj_type} is not in the TORCH_OPTIMIZERS'
    return getattr(torch.optim, obj_type)(**args)
    # torch.optim.AdamW(lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)

