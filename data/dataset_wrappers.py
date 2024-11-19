# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset


class ConcatDataset(_ConcatDataset):
    """A wrapper of concatenated dataset.

    Same as :obj:`torch.utils.data.dataset.ConcatDataset`, but
    自定义的 ConcatDataset 在初始化时从第一个数据集 datasets[0] 中复制了 CLASSES 属性。
    这假设所有数据集都使用相同的类别标签。torch.utils.data.dataset.ConcatDataset 本身没有这个属性。

    Args:
        datasets (list[:obj:`Dataset`]): A list of datasets.
    """

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        self.CLASSES = datasets[0].CLASSES
        self.PALETTE = datasets[0].PALETTE


class RepeatDataset(object):
    """A wrapper of repeated dataset.

    重复数据集的长度将比原始数据集大 “倍”。
    当数据加载时间较长但数据集较小时，此功能非常有用。
    使用 RepeatDataset 可以减少 epoch 之间的数据加载时间。

    Args:
        dataset (:obj:`Dataset`): The dataset to be repeated.
        times (int): Repeat times.
    """

    def __init__(self, dataset, times):
        self.dataset = dataset
        self.times = times
        self.CLASSES = dataset.CLASSES
        self.PALETTE = dataset.PALETTE
        self._ori_len = len(self.dataset)

    def __getitem__(self, idx):
        """Get item from original dataset."""
        return self.dataset[idx % self._ori_len]

    def __len__(self):
        """The length is multiplied by ``times``"""
        return self.times * self._ori_len
