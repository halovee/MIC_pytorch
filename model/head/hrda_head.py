from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from configs.cfg_model import head_cfg, get_cfg_model, attn_cfg
from model.base_module import BaseModule
from model.head.daformer_head import DAFormerHead

from model.head.decode_head import BaseDecodeHead
from model.tools.crossEntropyLoss import CrossEntropyLoss
from utils import add_prefix
from utils.wrappers import wrap_resize as _resize
from utils.losses.accuracy import accuracy


def scale_box(box, scale):
    y1, y2, x1, x2 = box
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2


class HRDAHead(BaseDecodeHead):
    '''结合了HRDAHead和BaseDecodeHead的内容'''

    def __init__(self,
                 single_scale_head='DAFormerHead',
                 lr_loss_weight=0,
                 hr_loss_weight=0.1,
                 scales=[0.5, 1],
                 attention_embed_dim=256,
                 attention_classwise=True,
                 enable_hr_crop=True,
                 hr_slide_inference=True,
                 fixed_attention=None,
                 **kwargs):
        head_cfg = deepcopy(kwargs)
        attn_cfg = deepcopy(kwargs)
        if single_scale_head == 'DAFormerHead':
            attn_cfg['channels'] = attention_embed_dim
            attn_cfg['decoder_params']['embed_dims'] = attention_embed_dim
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']['norm_cfg'])
            kwargs['init_cfg'] = None
            kwargs['input_transform'] = 'multiple_select'
            self.os = 4
        elif single_scale_head == 'DLV2Head':
            kwargs['init_cfg'] = None
            kwargs.pop('dilations')
            kwargs['channels'] = 1
            self.os = 8
        else:
            raise NotImplementedError(single_scale_head)
        super(HRDAHead, self).__init__(**kwargs)
        del self.conv_seg
        del self.dropout

        # HRDAHead中的构建
        self.head = DAFormerHead(**head_cfg)                       # segm.head

        if not attention_classwise:
            attn_cfg['num_classes'] = 1
        if fixed_attention is None:
            self.scale_attention = DAFormerHead(**attn_cfg)        # attentionHead
        else:
            self.scale_attention = None
            self.fixed_attention = fixed_attention
        self.lr_loss_weight = lr_loss_weight
        self.hr_loss_weight = hr_loss_weight
        self.scales = scales
        self.enable_hr_crop = enable_hr_crop
        self.hr_crop_box = None
        self.hr_slide_inference = hr_slide_inference

    def set_hr_crop_box(self, boxes):
        self.hr_crop_box = boxes

    def hr_crop_slice(self, scale):
        crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box, scale)
        return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)

    def resize(self, input, scale_factor):
        return _resize(
            input=input,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=self.align_corners)

    def decode_hr(self, inp, bs):
        if isinstance(inp, dict) and 'boxes' in inp.keys():
            features = inp['features']  # level, crop * bs, c, h, w
            boxes = inp['boxes']
            dev = features[0][0].device
            h_img, w_img = 0, 0
            for i in range(len(boxes)):
                boxes[i] = scale_box(boxes[i], self.os)
                y1, y2, x1, x2 = boxes[i]
                if h_img < y2:
                    h_img = y2
                if w_img < x2:
                    w_img = x2
            preds = torch.zeros((bs, self.num_classes, h_img, w_img), device=dev)
            count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)

            crop_seg_logits = self.head(features)
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1

            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            return preds
        else:
            return self.head(inp)

    def get_scale_attention(self, inp):
        if self.scale_attention is not None:
            att = torch.sigmoid(self.scale_attention(inp))
        else:
            att = self.fixed_attention
        return att

    def forward(self, inputs):
        assert len(inputs) == 2
        hr_inp = inputs[1]
        hr_scale = self.scales[1]
        lr_inp = inputs[0]
        lr_sc_att_inp = inputs[0]  # separate var necessary for stack hr_fusion
        lr_scale = self.scales[0]
        batch_size = lr_inp[0].shape[0]
        assert lr_scale <= hr_scale

        has_crop = self.hr_crop_box is not None
        if has_crop:
            crop_y1, crop_y2, crop_x1, crop_x2 = self.hr_crop_box

        # print_log(f'lr_inp {[f.shape for f in lr_inp]}', 'mmseg')
        lr_seg = self.head(lr_inp)
        # print_log(f'lr_seg {lr_seg.shape}', 'mmseg')

        hr_seg = self.decode_hr(hr_inp, batch_size)

        att = self.get_scale_attention(lr_sc_att_inp)
        if has_crop:
            mask = lr_seg.new_zeros([lr_seg.shape[0], 1, *lr_seg.shape[2:]])
            sc_os = self.os / lr_scale
            slc = self.hr_crop_slice(sc_os)
            mask[:, :, slc[0], slc[1]] = 1
            att = att * mask
        # print_log(f'att {att.shape}', 'mmseg')
        lr_seg = (1 - att) * lr_seg
        # print_log(f'scaled lr_seg {lr_seg.shape}', 'mmseg')
        up_lr_seg = self.resize(lr_seg, hr_scale / lr_scale)
        if torch.is_tensor(att):
            att = self.resize(att, hr_scale / lr_scale)

        if has_crop:
            hr_seg_inserted = torch.zeros_like(up_lr_seg)
            slc = self.hr_crop_slice(self.os)
            hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg
        else:
            hr_seg_inserted = hr_seg

        fused_seg = att * hr_seg_inserted + up_lr_seg

        return fused_seg, lr_seg, hr_seg

    def reset_crop(self):
        del self.hr_crop_box
        self.hr_crop_box = None

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, seg_weight=None, return_logits=False):
        """Forward function for training."""
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg, seg_weight)
        if return_logits:
            losses['logits'] = seg_logits
        self.reset_crop()
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing, only ``fused_seg`` is used."""
        return self.forward(inputs)[0]

    def losses(self, seg_logit, seg_label, seg_weight=None):
        """Compute losses."""
        from model.segmentors.hrdaEncodeDecode import crop
        fused_seg, lr_seg, hr_seg = seg_logit
        loss = super(HRDAHead, self).losses(fused_seg, seg_label, seg_weight)
        if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
            return loss

        if self.lr_loss_weight > 0:
            loss.update(add_prefix(super(HRDAHead, self).losses(lr_seg, seg_label, seg_weight), 'lr'))
        if self.hr_loss_weight > 0 and self.enable_hr_crop:
            cropped_seg_label = crop(seg_label, self.hr_crop_box)
            if seg_weight is not None:
                cropped_seg_weight = crop(seg_weight, self.hr_crop_box)
            else:
                cropped_seg_weight = seg_weight
            loss.update(add_prefix(super(HRDAHead, self).losses(hr_seg, cropped_seg_label, cropped_seg_weight), 'hr'))
        elif self.hr_loss_weight > 0:
            loss.update(add_prefix(super(HRDAHead, self).losses(hr_seg, seg_label, seg_weight), 'hr'))
        loss['loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)
        if self.lr_loss_weight > 0:
            loss['lr.loss_seg'] *= self.lr_loss_weight
        if self.hr_loss_weight > 0:
            loss['hr.loss_seg'] *= self.hr_loss_weight

        return loss

    # def losses(self, seg_logit, seg_label, seg_weight=None):
    #     """Compute losses."""
    #     from ..segmentors.hrdaEncodeDecode import crop
    #     fused_seg, lr_seg, hr_seg = seg_logit
    #     loss = self.super_losses(fused_seg, seg_label, seg_weight)
    #     if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
    #         return loss
    #
    #     if self.lr_loss_weight > 0:
    #         loss.update(add_prefix(self.super_losses(lr_seg, seg_label, seg_weight), 'lr'))
    #     if self.hr_loss_weight > 0 and self.enable_hr_crop:
    #         cropped_seg_label = crop(seg_label, self.hr_crop_box)
    #         if seg_weight is not None:
    #             cropped_seg_weight = crop(seg_weight, self.hr_crop_box)
    #         else:
    #             cropped_seg_weight = seg_weight
    #         loss.update(add_prefix(self.super_losses(hr_seg, cropped_seg_label, cropped_seg_weight), 'hr'))
    #     elif self.hr_loss_weight > 0:
    #         loss.update(add_prefix(self.super_losses(hr_seg, seg_label, seg_weight), 'hr'))
    #     loss['loss_seg'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)
    #     if self.lr_loss_weight > 0:
    #         loss['lr.loss_seg'] *= self.lr_loss_weight
    #     if self.hr_loss_weight > 0:
    #         loss['hr.loss_seg'] *= self.hr_loss_weight
    #     return loss
    #
    # def super_losses(self, seg_logit, seg_label, seg_weight=None):
    #     """Compute segmentation loss."""
    #     loss = dict()
    #     seg_logit = wrap_resize(input=seg_logit, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
    #     if self.sampler is not None:
    #         seg_weight = self.sampler.sample(seg_logit, seg_label)
    #     seg_label = seg_label.squeeze(1)
    #     loss['loss_seg'] = self.loss_decode(seg_logit, seg_label, weight=seg_weight, ignore_index=self.ignore_index)
    #     loss['acc_seg'] = accuracy(seg_logit, seg_label)
    #     return loss

    # def _init_inputs(self, in_channels, in_index, input_transform):
    #     """Check and initialize input transforms.        """
    #
    #     if input_transform is not None:
    #         assert input_transform in ['resize_concat', 'multiple_select']
    #     self.input_transform = input_transform
    #     self.in_index = in_index
    #     if input_transform is not None:
    #         assert isinstance(in_channels, (list, tuple))
    #         assert isinstance(in_index, (list, tuple))
    #         assert len(in_channels) == len(in_index)
    #         if input_transform == 'resize_concat':
    #             self.in_channels = sum(in_channels)
    #         else:
    #             self.in_channels = in_channels
    #     else:
    #         assert isinstance(in_channels, int)
    #         assert isinstance(in_index, int)
    #         self.in_channels = in_channels


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