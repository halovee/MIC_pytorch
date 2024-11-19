from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from configs.cfg_model import get_cfg_model
from model.backbone.mix_transformer import mit_b5
from model.base_module import BaseModule
from model.head.hrda_head import HRDAHead
from model.segmentors.base import BaseSegmentor
from utils import add_prefix
from utils import wrap_resize as _resize
import numpy as np

def get_crop_bbox(img_h, img_w, crop_size, divisible=1):
    """Randomly get a crop bounding box."""
    assert crop_size[0] > 0 and crop_size[1] > 0
    if img_h == crop_size[-2] and img_w == crop_size[-1]:
        return (0, img_h, 0, img_w)
    margin_h = max(img_h - crop_size[-2], 0)
    margin_w = max(img_w - crop_size[-1], 0)
    offset_h = np.random.randint(0, (margin_h + 1) // divisible) * divisible
    offset_w = np.random.randint(0, (margin_w + 1) // divisible) * divisible
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2


def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    if img.dim() == 4:
        img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 3:
        img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 2:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        raise NotImplementedError(img.dim())
    return img

class HRDAEncoderDecoder(BaseSegmentor):
    last_train_crop_box = {}

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,

                 scales=[1, 0.5],
                 hr_crop_size=(512, 512),
                 hr_slide_inference=True,
                 hr_slide_overlapping=True,
                 crop_coord_divisible=8,
                 blur_hr_crop=False,
                 feature_scale=0.5,
                 **kwargs):
        self.feature_scale_all_strs = ['all']
        if isinstance(feature_scale, str):
            assert feature_scale in self.feature_scale_all_strs
        scales = sorted(scales)
        decode_head['scales'] = scales
        decode_head['enable_hr_crop'] = hr_crop_size is not None
        decode_head['hr_slide_inference'] = hr_slide_inference
        super(HRDAEncoderDecoder, self).__init__(init_cfg=init_cfg)

        # EncoderDecoder中的构建
        if pretrained is not None:
            # assert backbone.get('pretrained') is None, 'both backbone and segmentor set pretrained weight'
            backbone['pretrained'] = pretrained
        self.backbone = mit_b5(**backbone)               # 构建了一个backbone，如果有pretrained参数，之后可以考虑加入
        if neck is not None:
            self.neck = neck
            print("无neck的实现方法")

        self.decode_head = HRDAHead(**decode_head)                     # 构建了一个 HRDAHead
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        if auxiliary_head:
            self.auxiliary_head = auxiliary_head
            print("无auxiliary_head的实现方法")

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.local_iter = 0

        # HRDAEncoderDecoder中的构建
        self.scales = scales
        self.feature_scale = feature_scale
        self.crop_size = hr_crop_size
        self.hr_slide_inference = hr_slide_inference
        self.hr_slide_overlapping = hr_slide_overlapping
        self.crop_coord_divisible = crop_coord_divisible
        self.blur_hr_crop = blur_hr_crop


    def extract_unscaled_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_slide_feat(self, img):
        if self.hr_slide_overlapping:
            h_stride, w_stride = [e // 2 for e in self.crop_size]
        else:
            h_stride, w_stride = self.crop_size
        h_crop, w_crop = self.crop_size
        bs, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        crop_imgs, crop_feats, crop_boxes = [], [], []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_imgs.append(img[:, :, y1:y2, x1:x2])
                crop_boxes.append([y1, y2, x1, x2])
        crop_imgs = torch.cat(crop_imgs, dim=0)
        crop_feats = self.extract_unscaled_feat(crop_imgs)
        # shape: feature levels, crops * batch size x c x h x w

        return {'features': crop_feats, 'boxes': crop_boxes}

    def blur_downup(self, img, s=0.5):
        img = _resize(
            input=img,
            scale_factor=s,
            mode='bilinear',
            align_corners=self.align_corners)
        img = _resize(
            input=img,
            scale_factor=1 / s,
            mode='bilinear',
            align_corners=self.align_corners)
        return img

    def resize(self, img, s):
        if s == 1:
            return img
        else:
            with torch.no_grad():
                return _resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)

    def extract_feat(self, img):
        if self.feature_scale in self.feature_scale_all_strs:
            mres_feats = []
            for i, s in enumerate(self.scales):
                if s == 1 and self.blur_hr_crop:
                    scaled_img = self.blur_downup(img)
                else:
                    scaled_img = self.resize(img, s)
                if self.crop_size is not None and i >= 1:
                    scaled_img = crop(scaled_img, HRDAEncoderDecoder.last_train_crop_box[i])
                mres_feats.append(self.extract_unscaled_feat(scaled_img))
            return mres_feats
        else:
            scaled_img = self.resize(img, self.feature_scale)
            return self.extract_unscaled_feat(scaled_img)

    def generate_pseudo_label(self, img, img_metas):
        out = self.encode_decode(img, img_metas)
        return out

    def encode_decode(self, img, img_metas, upscale_pred=True):
        """Encode images with backbone and decode into a semantic segmentation map of the same size as input."""
        mres_feats = []
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = self.resize(img, s)
            if i >= 1 and self.hr_slide_inference:
                mres_feats.append(self.extract_slide_feat(scaled_img))
            else:
                mres_feats.append(self.extract_unscaled_feat(scaled_img))
            # if self.decode_head.debug:
            #     self.decode_head.debug_output[f'Img {i} Scale {s}'] = scaled_img.detach()
        out = self._decode_head_forward_test(mres_feats, img_metas)
        if upscale_pred:
            out = _resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out

    def _forward_train_features(self, img):
        mres_feats = []
        # self.decode_head.debug_output = {}
        assert len(self.scales) <= 2, 'Only up to 2 scales are supported.'
        prob_vis = None
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = _resize(input=img, scale_factor=s, mode='bilinear', align_corners=self.align_corners)
            if self.crop_size is not None and i >= 1:
                crop_box = get_crop_bbox(*scaled_img.shape[-2:], self.crop_size, self.crop_coord_divisible)
                if self.feature_scale in self.feature_scale_all_strs:
                    HRDAEncoderDecoder.last_train_crop_box[i] = crop_box
                self.decode_head.set_hr_crop_box(crop_box)
                scaled_img = crop(scaled_img, crop_box)
            # if self.decode_head.debug:
            #     self.decode_head.debug_output[f'Img {i} Scale {s}'] = scaled_img.detach()
            mres_feats.append(self.extract_unscaled_feat(scaled_img))
        return mres_feats, prob_vis

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      seg_weight=None,
                      return_feat=False,
                      return_logits=False):
        """Forward function for training.
        """
        # self.update_debug_state()

        losses = dict()

        mres_feats, prob_vis = self._forward_train_features(img)       # 下次核实一下是否有两种尺度的特征图，即s=0.5和1的这两种
        for i, s in enumerate(self.scales):
            if return_feat and self.feature_scale in self.feature_scale_all_strs:
                if 'features' not in losses:
                    losses['features'] = []
                losses['features'].append(mres_feats[i])
            if return_feat and s == self.feature_scale:                # 这里返回的特征图，仅有s=0.5的这一种，而没有s=1的这一种
                losses['features'] = mres_feats[i]
                break
        # 传入的mres_feats含有两个inp, inp[1]是1倍尺度的hr_inp，inp[0]是0.5倍尺度的lr_inp，核实一下是否是这样！！！
        loss_decode = self._decode_head_forward_train(mres_feats, img_metas, gt_semantic_seg, seg_weight, return_logits)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            raise NotImplementedError

        # if self.with_auxiliary_head:
        #     loss_aux = self._auxiliary_head_forward_train(mres_feats, img_metas, gt_semantic_seg, seg_weight)
        #     losses.update(loss_aux)

        self.local_iter += 1
        return losses

    def forward_with_aux(self, img, img_metas):
        assert not self.with_auxiliary_head
        mres_feats, _ = self._forward_train_features(img)
        out = self.decode_head.forward(mres_feats)
        # out = wrap_resize(
        #     input=out,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return {'main': out}


    # -----------------------------------添加自encoder_decoder的部分-----------------------------------

    def _decode_head_forward_train(self,
                                   x,
                                   img_metas,
                                   gt_semantic_seg,
                                   seg_weight=None,
                                   return_logits=False):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        # 进入hrda_head的forward_train函数
        loss_decode = self.decode_head.forward_train(x, img_metas, gt_semantic_seg, self.train_cfg, seg_weight, return_logits)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits


    # -----------------------------------添加自encoder_decoder的部分-----------------------------------

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):           # encoder_decoder的部分
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg['stride']
        h_crop, w_crop = self.test_cfg['crop_size']
        batched_slide = self.test_cfg.get('batched_slide', False)
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        if batched_slide:
            crop_imgs, crops = [], []
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_imgs.append(crop_img)
                    crops.append((y1, y2, x1, x2))
            crop_imgs = torch.cat(crop_imgs, dim=0)
            crop_seg_logits = self.encode_decode(crop_imgs, img_meta)
            for i in range(len(crops)):
                y1, y2, x1, x2 = crops[i]
                crop_seg_logit = crop_seg_logits[i * batch_size:(i + 1) * batch_size]
                preds += F.pad(crop_seg_logit,(int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        else:
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_seg_logit = self.encode_decode(crop_img, img_meta)
                    preds += F.pad(crop_seg_logit,(int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
                    count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = _resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = _resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):        # encoder_decoder的部分
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg['mode'] in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg['mode'] == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        if hasattr(self.decode_head, 'debug_output_attention') and self.decode_head.debug_output_attention:
            output = seg_logit
        else:
            output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):            # encoder_decoder的部分
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if hasattr(self.decode_head, 'debug_output_attention') and self.decode_head.debug_output_attention:
            seg_pred = seg_logit[:, 0]
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    # def init_weights(self):
    #     """Initialize the weights."""
    #     import warnings
    #     from collections import defaultdict
    #     from utils.weight_init import initialize, update_init_info
    #     is_top_level_module = False
    #     if not hasattr(self, '_params_init_info'):
    #         self._params_init_info = defaultdict(dict)
    #         is_top_level_module = True
    #         for name, param in self.named_parameters():
    #             self._params_init_info[param]['init_info'] = f'The value is the same before and after calling `init_weights` of {self.__class__.__name__} '
    #             self._params_init_info[param]['tmp_mean_value'] = param.data.mean()
    #         for sub_module in self.modules():
    #             sub_module._params_init_info = self._params_init_info
    #     module_name = self.__class__.__name__
    #     if not self.is_init:
    #         if self.init_cfg:
    #             print(f'initialize {module_name} with init_cfg {self.init_cfg}')
    #             initialize(self, self.init_cfg)
    #             if isinstance(self.init_cfg, dict):
    #                 if self.init_cfg['type'] == 'Pretrained':
    #                     return
    #         for m in self.children():
    #             if hasattr(m, 'init_weights'):
    #                 m.init_weights()
    #                 update_init_info(m, init_info=f'Initialized by user-defined `init_weights` in {m.__class__.__name__} ')
    #         self.is_init = True
    #     else:
    #         warnings.warn(f'init_weights of {self.__class__.__name__} has been called more than once.')
    #     if is_top_level_module:
    #         for sub_module in self.modules():
    #             del sub_module._params_init_info

    # def whole_inference(self, img, img_meta, rescale):
    #     """Inference with full image."""
    #
    #     seg_logit = self.encode_decode(img, img_meta)
    #     if rescale:
    #         # support dynamic shape for onnx
    #         if torch.onnx.is_in_onnx_export():
    #             size = img.shape[2:]
    #         else:
    #             size = img_meta[0]['ori_shape'][:2]
    #         seg_logit = wrap_resize(
    #             seg_logit,
    #             size=size,
    #             mode='bilinear',
    #             align_corners=self.align_corners,
    #             warning=False)
    #
    #     return seg_logit
    #
    #
    #
    # def aug_test(self, imgs, img_metas, rescale=True):
    #     """Test with augmentations.
    #
    #     Only rescale=True is supported.
    #     """
    #     # aug_test rescale all imgs back to ori_shape for now
    #     assert rescale
    #     # to save memory, we get augmented seg logit inplace
    #     seg_logit = self.inference(imgs[0], img_metas[0], rescale)
    #     for i in range(1, len(imgs)):
    #         cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
    #         seg_logit += cur_seg_logit
    #     seg_logit /= len(imgs)
    #     seg_pred = seg_logit.argmax(dim=1)
    #     seg_pred = seg_pred.cpu().numpy()
    #     # unravel batch dim
    #     seg_pred = list(seg_pred)
    #     return seg_pred
