import os
import random

import numpy as np
import torch


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, inner_iter, learning_rate=None, max_iters=None, lr_power=None):
    lr = lr_poly(learning_rate, inner_iter, max_iters, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1 :
        optimizer.param_groups[1]['lr'] = lr * 10


class Learning_Rate_Object(object):
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool):
            是否为 CUDNN 后端设置 deterministic 选项，
            即，将 'torch.backends.cudnn.deterministic' 设置为 True，
            将 'torch.backends.cudnn.benchmark' 设置为 False。
            Default: False.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

'''
torch.backends.cudnn.deterministic 和 torch.backends.cudnn.benchmark 是
两个与PyTorch中CuDNN（Convolutional Neural Network library）相关的设置，它们影响卷积操作的性能和确定性。

torch.backends.cudnn.deterministic
当设置 torch.backends.cudnn.deterministic = True 时，CuDNN 会选择确定性算法来执行卷积操作。
这意味着每次运行相同的代码时，都会得到相同的结果。这对于调试和确保实验的可重复性非常有用。
    优点：确保每次运行都能得到相同的结果，便于复现和调试。
    缺点：可能会牺牲一些性能，因为确定性算法可能不是最快的。

torch.backends.cudnn.benchmark
当设置 torch.backends.cudnn.benchmark = True 时，
CuDNN 会进行一系列运行来寻找最适合当前配置（例如，网络结构和输入数据大小）的算法，并缓存这些选择以供后续使用。
这通常可以显著提高卷积网络的性能。
    优点：通过选择最优算法，可以显著提高卷积操作的运行速度。
    缺点：由于每次运行可能会选择不同的算法，这可能会导致结果的不确定性，使得实验难以复现。
    
    =================
    
    使用建议
    在调试或复现实验时：设置 torch.backends.cudnn.deterministic = True。
    在生产或训练模型时，且对性能有高要求：设置 torch.backends.cudnn.benchmark = True，但要注意，这可能使得结果在不同运行之间不可复现。

    通常，你不会同时将这两个选项都设置为 True，
    因为 benchmark 的目的是优化性能，而 deterministic 的目的是确保结果的可复现性，
    这两者的目标在一定程度上是相互冲突的。
'''