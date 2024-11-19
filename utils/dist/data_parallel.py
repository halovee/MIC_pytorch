
from itertools import chain
from torch.nn.parallel import DataParallel


class MMDataParallel(DataParallel):
    """MMDataParallel模块支持DataContainer。
    MMDataParallel与PyTorch DataParallel主要有以下两个区别：
        它支持自定义类型：class:DataContainer，这使得在GPU和CPU推理过程中对输入数据的控制更加灵活。
        它实现了两个额外的API train_step() 和 val_step()。
    .. warning::
        MMDataParallel仅支持单GPU训练，如果您需要使用多个GPU进行训练，请使用MMDistributedDataParallel。
        如果您有多个GPU但只想使用MMDataParallel，您可以设置环境变量
        CUDA_VISIBLE_DEVICES=0 或者用 device_ids=[0] 实例化 MMDataParallel。
    参数：
        module (:class:nn.Module)：要封装的模块。
        device_ids (list[int])：要将模块分散到的设备ID。当GPU不可用时，默认为None。
        output_device (str | int)：输出设备ID。默认为None。
        dim (int)：用于分散数据的维度。默认为0。
    """

    def __init__(self, *args, dim=0, **kwargs):
        super(MMDataParallel, self).__init__(*args, dim=dim, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        """Override the original forward function.

        The main difference lies in the CPU inference where the data in
        :class:`DataContainers` will still be gathered.
        """
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module(*inputs[0], **kwargs[0])
        else:
            return super().forward(*inputs, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        from ..dist import scatter_kwargs
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def train_step(self, *inputs, **kwargs):
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.train_step(*inputs[0], **kwargs[0])

        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.train_step(*inputs[0], **kwargs[0])

    def val_step(self, *inputs, **kwargs):
        if not self.device_ids:
            # We add the following line thus the module could gather and
            # convert data containers as those in GPU inference
            inputs, kwargs = self.scatter(inputs, kwargs, [-1])
            return self.module.val_step(*inputs[0], **kwargs[0])

        assert len(self.device_ids) == 1, \
            ('MMDataParallel only supports single GPU training, if you need to'
             ' train with multiple GPUs, please use MMDistributedDataParallel'
             ' instead.')

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers '
                    f'on device {self.src_device_obj} (device_ids[0]) but '
                    f'found one of them on device: {t.device}')

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.val_step(*inputs[0], **kwargs[0])
