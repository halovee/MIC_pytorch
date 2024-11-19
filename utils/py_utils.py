from collections import abc
import collections.abc
from itertools import repeat
from pathlib import Path

def is_module_wrapper(module):
    """Check if a module is a module wrapper.
    """
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    MODULE_WRAPPERS = DataParallel, DistributedDataParallel
    module_wrappers = tuple(MODULE_WRAPPERS)
    return isinstance(module, module_wrappers)

# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)

def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)

def has_method(obj: object, method: str) -> bool:
    """Check whether the object has a method.

    Args:
        method (str): The method name to check.
        obj (object): The object to check.

    Returns:
        bool: True if the object has the method else False.
    """
    return hasattr(obj, method) and callable(getattr(obj, method))

def is_filepath(x):
    return isinstance(x, str) or isinstance(x, Path)