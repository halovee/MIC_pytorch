import collections
import os.path as osp
import random
import warnings

import numpy as np
from utils.filesio import FileClient
from utils.image import bgr2hsv, hsv2bgr, imfrombytes, imrescale, imresize, imflip, impad, impad_to_multiple, imnormalize
from utils.py_utils import is_list_of
from  .formating import to_tensor, DataContainer, ToDataContainer, Transpose , ImageToTensor, ToTensor


class Compose(object):
    """Compose multiple transforms sequentially.
    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_dataaug(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.
        与torchvision.transforms.Compose的__call__方法一样，对data进行一系列的变换
        Args:
            data (dict): A result dict contains the data to transform.
        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        '''与torchvision.transforms.Compose的__repr__方法一样，返回一个字符串，表示Compose对象'''
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string



class LoadImageFromFile(object):
    """从文件中加载图像。
    必需的键是 "img_prefix" 和 "img_info"（一个必须包含键 "filename" 的字典）。
    添加或更新的键包括 "filename"、"img"、
                    "img_shape"、"ori_shape"（与 `img_shape` 相同）、"pad_shape"（与 `img_shape` 相同）、
                    "scale_factor"（值为 1.0）以及 "img_norm_cfg"（均值为 0，标准差为 1）。
    参数：
    - to_float32 (bool)：是否将加载的图像转换为 float32 类型的 numpy 数组。如果设置为 False，则加载的图像是 uint8 类型的数组。默认为 False。
    - color_type (str)：传递给 :func:`mmcv.imfrombytes` 函数的标志参数。默认为 'color'。
    - file_client_args (dict)：用于实例化 FileClient 的参数。详情请参见 :class:`mmcv.fileio.FileClient`。默认为 ``dict(backend='disk')``。
    - imdecode_backend (str)：:func:`mmcv.imdecode` 的后端。默认为 'cv2'。

    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """调用函数加载图像并获取图像元信息。
        参数：
            results （dict）：来自 ：obj：'mmseg 的结果 dict。CustomDataset“的 Dataset 中。
        返回：
            dict：该 dict 包含加载的图片和元信息。
        """

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                 reduce_zero_label=False,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.
        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.
        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        img_bytes = self.file_client.get(filename)
        gt_semantic_seg = imfrombytes(img_bytes, flag='unchanged', backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


class Resize(object):
    """Resize images & seg.
    这个变换将输入图像调整到某个尺度。如果输入字典包含键 "scale"，则使用输入字典中的尺度，否则使用初始化方法中指定的尺度。
    ``img_scale`` 可以是 None，一个元组（单尺度）或一个元组列表（多尺度）。有4种多尺度模式：
    - ``ratio_range 不为 None``：
      1. 当 img_scale 为 None 时，img_scale 是结果中图像的形状（img_scale = results['img'].shape[:2]），并且图像基于原始大小进行调整。（模式 1）
      2. 当 img_scale 是一个元组（单尺度）时，从比例范围中随机采样一个比例，并将其与图像尺度相乘。（模式 2）
    - ``ratio_range 为 None 且 multiscale_mode == "range"``：从范围内随机采样一个尺度。（模式 3）
    - ``ratio_range 为 None 且 multiscale_mode == "value"``：从多个尺度中随机采样一个尺度。（模式 4）
    参数：
    - img_scale (tuple 或 list[tuple])：用于调整图像大小的图像尺度。默认：None。
    - multiscale_mode (str)："range" 或 "value"。默认：'range'
    - ratio_range (tuple[float])：(最小比例, 最大比例)。默认：None
    - keep_ratio (bool)：调整图像大小时是否保持宽高比。默认：True
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 override_scale=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: 给定多个尺度或一系列尺度
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.override_scale = override_scale

    @staticmethod
    def random_select(img_scales):
        """从给定的候选中选择一个随机的 img_scale。
        参数：
        - img_scales (list[tuple])：用于选择的图像尺度列表。
        返回：
        - (tuple, int)：返回一个元组 ``(img_scale, scale_idx)`，其中 ``img_scale`` 是选中的图像尺度，``scale_idx`` 是在给定候选中的选中索引。
        """
        assert is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """当 ``multiscale_mode=='range'`` 时，从 ``img_scales`` 中随机采样一个 ``img_scale``。
        参数：
        - img_scales (list[tuple])：用于采样的图像尺度范围。``img_scales`` 中必须有两个元组，它们指定图像尺度的下界和上界。
        返回：
        - (tuple, None)：返回一个元组 ``(img_scale, None)`，
            其中 ``img_scale`` 是采样的尺度，而 ``None`` 只是一个占位符，用于与 :func:`random_select` 保持一致。
        """
        assert is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """当 ``ratio_range`` 被指定时，从 ``ratio_range`` 指定的范围内随机采样一个 ``img_scale``。
        一个比例将从 ``ratio_range`` 指定的范围内随机采样。然后它将与 ``img_scale`` 相乘，以生成采样的尺度。
        参数：
        - img_scale (tuple)：与比例相乘的图像尺度基础。
        - ratio_range (tuple[float])：用于缩放 ``img_scale`` 的最小和最大比例。
        返回：
        - (tuple, None)：返回一个元组 ``(scale, None)`，其中 ``scale`` 是与 ``img_scale`` 相乘的采样比例，而 ``None`` 只是一个占位符，用于与 :func:`random_select` 保持一致。

        """
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """根据 ``ratio_range`` 和 ``multiscale_mode`` 随机采样一个 ``img_scale``。
        如果指定了 ``ratio_range``，将采样一个比例并与 ``img_scale`` 相乘。
        如果 ``img_scale`` 指定了多个尺度，将根据 ``multiscale_mode`` 采样一个尺度。
        否则，将使用单一尺度。
        参数：
        - results (dict)：来自 :obj:`dataset` 的结果字典。
        返回：
        - dict：两个新键 'scale` 和 'scale_idx` 被添加到 ``results`` 中，这将由后续的管道使用。
        """
        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h), self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = imrescale(results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = imresize(results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = imrescale(results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = imresize(results[key], results['scale'], interpolation='nearest')
            results[key] = gt_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', 'keep_ratio' keys are added into result dict.
        """
        if 'scale' not in results or self.override_scale:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


class RandomFlip(object):
    """翻转图像和分割图。
    如果输入字典包含键 "flip"，则将使用该标志，否则它将根据初始化方法中指定的比例随机决定。
    参数：
    - prob (float, 可选)：翻转概率。默认：None。
    - direction (str, 可选)：翻转方向。选项有'horizontal'（水平）和 'vertical'（垂直）。默认：'horizontal'（水平）。
    """

    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into result dict.
        """
        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = imflip(results['img'], direction=results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = imflip(results[key], direction=results['flip_direction']).copy()
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'


class PhotoMetricDistortion(object):
    """对图像顺序应用光度失真，每个变换应用的概率为0.5。随机对比度的位置在第二个或倒数第二个。
        1. 随机亮度
        2. 随机对比度（模式0）
        3. 将颜色从BGR转换为HSV
        4. 随机饱和度
        5. 随机色调
        6. 将颜色从HSV转换回BGR
        7. 随机对比度（模式1）
        参数：
        - brightness_delta (int): 亮度的变化量。
        - contrast_range (tuple): 对比度的范围。
        - saturation_range (tuple): 饱和度的范围。
        - hue_delta (int): 色调的变化量。
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(img, alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower, self.saturation_upper))
            img = hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) + random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.      """

        img = results['img']
        # random brightness
        img = self.brightness(img)
        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)
        # random saturation
        img = self.saturation(img)
        # random hue
        img = self.hue(img)
        # random contrast
        if mode == 0:
            img = self.contrast(img)
        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


class Normalize(object):
    """Normalize the image.  """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.  """

        results['img'] = imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb='f'{self.to_rgb})'
        return repr_str


class Pad(object):
    """Pad the image & mask. """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = impad(results['img'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = impad_to_multiple(results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = impad(results[key], shape=results['pad_shape'][:2], pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps. """

        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, 'f'pad_val={self.pad_val})'
        return repr_str

class MultiScaleFlipAug(object):
    """Test-time augmentation with multiple scales and flipping.

    Args:
        transforms (list[dict]): 在每个增强中要应用的变换。
        img_scale (None | tuple | list[tuple]): 用于调整大小的图像尺度。
        img_ratios (float | list[float]): 用于调整大小的图像比率。
        flip (bool): 是否应用翻转增强。默认：False。
        flip_direction (str | list[str]): Flip 翻转增强的方向，选项有 “horizontal”（水平）和 “vertical”（垂直）。
            如果 flip_direction 是列表，将应用多个翻转增强。 当 flip == False 时无效。默认：“horizontal”。
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 img_ratios=None,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose(transforms)
        if img_ratios is not None:
            img_ratios = img_ratios if isinstance(img_ratios, list) else [img_ratios]
            assert is_list_of(img_ratios, float)
        if img_scale is None:
            # mode 1: given img_scale=None and a range of image ratio
            self.img_scale = None
            assert is_list_of(img_ratios, float)
        elif isinstance(img_scale, tuple) and is_list_of(img_ratios, float):
            assert len(img_scale) == 2
            # mode 2: given a scale and a range of image ratio
            self.img_scale = [(int(img_scale[0] * ratio), int(img_scale[1] * ratio)) for ratio in img_ratios]
        else:
            # mode 3: given multiple scales
            self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
        assert is_list_of(self.img_scale, tuple) or self.img_scale is None
        self.flip = flip
        self.img_ratios = img_ratios
        self.flip_direction = flip_direction if isinstance(flip_direction, list) else [flip_direction]
        assert is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn('flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn('flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        aug_data = []
        if self.img_scale is None and is_list_of(self.img_ratios, float):
            h, w = results['img'].shape[:2]
            img_scale = [(int(w * ratio), int(h * ratio)) for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    _results = results.copy()
                    _results['scale'] = scale
                    _results['flip'] = flip
                    _results['flip_direction'] = direction
                    data = self.transforms(_results)
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip})'
        repr_str += f'flip_direction={self.flip_direction}'
        return repr_str

class DefaultFormatBundle(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor, (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.
        Args:
            results (dict): Result dict 包含要转换的数据。
        Returns:
            dict: The result dict contains the data that is formatted with default bundle.
        """

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DataContainer(to_tensor(img), stack=True)

        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DataContainer(to_tensor(results['gt_semantic_seg'][None, ...].astype(np.int64)), stack=True)

        if 'valid_pseudo_mask' in results:
            results['valid_pseudo_mask'] = DataContainer(to_tensor(results['valid_pseudo_mask'][None, ...]), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


class Collect(object):
    """Collect data from the loader relevant to the specific task.
    This is usually the last stage of the data loader pipeline.
    Typically keys is set to some subset of "img", "gt_semantic_seg".
    The "img_meta" item is always populated.
    The contents of the "img_meta" dictionary depends on "meta_keys". By default this includes:

        - "img_shape": shape of the image input to the network as a tuple (h, w, c).
            Note that images may be zero padded on the bottom/right if the batch tensor is larger than this shape.
        - "scale_factor": a float indicating the preprocessing scale
        - "flip": a boolean indicating if image flip transform was used
        - "filename": path to the image file
        - "ori_shape": original shape of the image as a tuple (h, w, c)
        - "pad_shape": image shape after padding
        - "img_norm_cfg": a dict of normalization information:
            - mean - per channel mean subtraction
            - std - per channel std divisor
            - to_rgb - bool indicating if bgr was converted to rgb

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional):
        元键被转换为 ''DataContainer'' 并收集在 ''data[img_metas]'' 中。
        默认值： ''（'filename'， 'ori_filename'， 'ori_shape'， 'img_shape'， 'pad_shape'， 'scale_factor'，'flip'， 'flip_direction'， 'img_norm_cfg'）''    """

    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results.
        The keys in ``meta_keys`` will be converted to :obj:mmcv.DataContainer.
        Args:
            results (dict): Result dict 包含要收集的数据。
        Returns:
            dict: The result dict contains the following keys
                - keys in``self.keys``
                - ``img_metas``
        """
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DataContainer(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys}, meta_keys={self.meta_keys})'

PIPELINES = {
    'Compose': Compose,
    'LoadImageFromFile': LoadImageFromFile,
    'LoadAnnotations': LoadAnnotations,
    'Resize': Resize,
    'RandomFlip': RandomFlip,
    'RandomCrop': RandomCrop,
    'Normalize': Normalize,
    'Pad': Pad,
    'PhotoMetricDistortion': PhotoMetricDistortion,
    'MultiScaleFlipAug': MultiScaleFlipAug,
    'Collect': Collect,
    'DefaultFormatBundle': DefaultFormatBundle,
    'ImageToTensor': ImageToTensor,
    'ToTensor': ToTensor,
    'ToDataContainer': ToDataContainer,
    'Transpose': Transpose,
}

def get_Class(obj_type):
    return PIPELINES[obj_type]

def build_dataaug(cfg):
    args = cfg.copy()
    obj_type = args.pop('type')
    if obj_type not in PIPELINES:
        raise KeyError(f'Type {obj_type} is not in the PIPELINES')
    obj_cls = get_Class(obj_type)
    return  obj_cls(**args)
