# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Save label with train_id color map if opacity==1

import os
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from model.base_module import BaseModule
from model.basesegmentor_wrappers import auto_fp16
from utils import mkdir_or_exist
from utils.image import imread, imwrite
from utils.image.image import imshow


class BaseSegmentor(BaseModule, metaclass=ABCMeta):
    """Base class for segmentors."""

    def __init__(self, init_cfg=None):
        super(BaseSegmentor, self).__init__(init_cfg)
        self.fp16_enabled = False

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self,
                       'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_decode_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @abstractmethod
    def extract_feat(self, imgs):
        """Placeholder for extract features from images."""
        pass

    @abstractmethod
    def encode_decode(self, img, img_metas, upscale_pred=True):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        pass

    @abstractmethod
    def forward_train(self, imgs, img_metas, **kwargs):
        """用于训练的 Forward 函数的占位符."""
        pass

    @abstractmethod
    def simple_test(self, img, img_meta, **kwargs):
        """Placeholder for single image test."""
        pass

    @abstractmethod
    def aug_test(self, imgs, img_metas, **kwargs):
        """Placeholder for augmentation test."""
        pass

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time augmentations and inner Tensor should have a shape NxCxHxW, which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time augs (multiscale, flip, etc.) and the inner list indicates images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """根据 `return_loss` 参数的值来调用 `forward_train` 或 `forward_test` 方法。

        注意这种设置将会改变期望的输入格式。
        当 `return_loss=True` 时，`img` 和 `img_meta` 是单层嵌套的（即，`Tensor` 和 `List[dict]`），
        而当 `return_loss=False` 时，`img` 和 `img_meta` 应该是双层嵌套的（即，`List[Tensor]`, `List[List[dict]]`），
        外层列表表示测试时的数据增强。
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def train_step(self, data_batch, optimizer, **kwargs):
        """训练过程中的一个迭代步骤。

        此方法定义了训练过程中的一个迭代步骤，除了反向传播和优化器更新之外，这些通常由优化器钩子处理。
        需要注意的是，在某些复杂的情况或模型中，整个过程包括反向传播和优化器更新也可能在这个方法中定义，例如生成对抗网络（GAN）。

        参数:
        - `data (dict)`: 数据加载器的输出。
        - `optimizer (:obj:`torch.optim.Optimizer` | dict)`: 运行器传递给 `train_step()` 的优化器。此参数未使用并保留。

        返回:
            - `dict`: 至少应该包含 3 个键：`loss`、`log_vars` 和 `num_samples`。
            - `loss` 是一个用于反向传播的张量，它可以是多个损失的加权和。
            - `log_vars` 包含所有要发送给日志记录器的变量。
            - `num_samples` 表示批次大小（当模型是分布式数据并行（DDP）时，它指的是每个 GPU 上的批次大小），用于计算日志的平均值。
        """
        losses = self(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data_batch['img_metas']))

        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        output = self(**data_batch, **kwargs)
        return output

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = imread(img)
        img = img.copy()
        seg = result[0]
        if palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(self.CLASSES), 3))
            else:
                palette = self.PALETTE
        palette = np.array(palette)
        assert palette.shape[0] == len(self.CLASSES)
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0

        # Save label with train_id color map
        if out_file is not None and opacity == 1.0:
            palette = np.array(self.PALETTE, dtype=np.uint8)
            out = Image.fromarray(np.array(seg).astype(np.uint8)).convert('P')
            out.putpalette(palette)
            mkdir_or_exist(os.path.abspath(os.path.dirname(out_file)))
            out.save(out_file)
            return

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False

        if show:
            imshow(img, win_name, wait_time)
        if out_file is not None:
            imwrite(img, out_file)

        if not (show or out_file):
            warnings.warn('show==False and out_file is not specified, only result image will be returned')
            return img
