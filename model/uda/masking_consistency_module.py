# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import random

import torch
from torch.nn import Module

from model.uda.dacs_transforms import strong_transform, get_mean_std
from utils.wrappers import wrap_resize


def build_mask_generator(cfg):
    if cfg is None:
        return None
    t = cfg.pop('type')
    if t == 'block':
        return BlockMaskGenerator(**cfg)
    else:
        raise NotImplementedError(t)


class BlockMaskGenerator:

    def __init__(self, mask_ratio, mask_block_size):
        self.mask_ratio = mask_ratio
        self.mask_block_size = mask_block_size

    @torch.no_grad()
    def generate_mask(self, imgs):
        B, _, H, W = imgs.shape

        mshape = B, 1, round(H / self.mask_block_size), round(W / self.mask_block_size)
        input_mask = torch.rand(mshape, device=imgs.device)
        input_mask = (input_mask > self.mask_ratio).float()
        input_mask = wrap_resize(input_mask, size=(H, W))
        return input_mask

    @torch.no_grad()
    def mask_image(self, imgs):
        input_mask = self.generate_mask(imgs)
        return imgs * input_mask


class MaskingConsistencyModule(Module):

    def __init__(self, require_teacher, cfg):
        super(MaskingConsistencyModule, self).__init__()

        self.source_only = cfg.get('source_only', False)
        self.max_iters = cfg['max_iters']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']

        self.mask_mode = cfg['mask_mode']
        self.mask_alpha = cfg['mask_alpha']
        self.mask_pseudo_threshold = cfg['mask_pseudo_threshold']
        self.mask_lambda = cfg['mask_lambda']
        self.mask_gen = build_mask_generator(cfg['mask_generator'])

        assert self.mask_mode in [
            'separate', 'separatesrc', 'separatetrg', 'separateaug',
            'separatesrcaug', 'separatetrgaug'
        ]

        self.teacher = None
        # if require_teacher or self.mask_alpha != 'same' or self.mask_pseudo_threshold != 'same':
        #     self.teacher = EMATeacher(use_mask_params=True, cfg=cfg)

        # self.debug = False
        # self.debug_output = {}

    def update_weights(self, model, iter):
        if self.teacher is not None:
            self.teacher.update_weights(model, iter)

    # def update_debug_state(self):
    #     if self.teacher is not None:
    #         self.teacher.debug = self.debug

    def __call__(self,
                 model,
                 img,
                 img_metas,
                 gt_semantic_seg,
                 target_img,
                 target_img_metas,
                 valid_pseudo_mask,
                 pseudo_label=None,
                 pseudo_weight=None):
        # self.update_debug_state()
        # self.debug_output = {}
        # model.debug_output = {}
        dev = img.device
        means, stds = get_mean_std(img_metas, dev)

        if not self.source_only:
            # Share the pseudo labels with the host UDA method
            if self.teacher is None:
                assert self.mask_alpha == 'same'
                assert self.mask_pseudo_threshold == 'same'
                assert pseudo_label is not None
                assert pseudo_weight is not None
                masked_plabel = pseudo_label
                masked_pweight = pseudo_weight
            # Use a separate EMA teacher for MIC
            else:
                masked_plabel, masked_pweight = \
                    self.teacher(target_img, target_img_metas, valid_pseudo_mask)
                # if self.debug:
                #     self.debug_output['Mask Teacher'] = {
                #         'Img': target_img.detach(),
                #         'Pseudo Label': masked_plabel.cpu().numpy(),
                #         'Pseudo Weight': masked_pweight.cpu().numpy(),
                #     }
        # Don't use target images at all
        if self.source_only:
            masked_img = img
            masked_lbl = gt_semantic_seg
            b, _, h, w = gt_semantic_seg.shape
            masked_seg_weight = None
        # Use 1x source image and 1x target image for MIC
        elif self.mask_mode in ['separate', 'separateaug']:
            assert img.shape[0] == 2
            masked_img = torch.stack([img[0], target_img[0]])
            masked_lbl = torch.stack(
                [gt_semantic_seg[0], masked_plabel[0].unsqueeze(0)])
            gt_pixel_weight = torch.ones(masked_pweight[0].shape, device=dev)
            masked_seg_weight = torch.stack(
                [gt_pixel_weight, masked_pweight[0]])
        # Use only source images for MIC
        elif self.mask_mode in ['separatesrc', 'separatesrcaug']:
            masked_img = img
            masked_lbl = gt_semantic_seg
            masked_seg_weight = None
        # Use only target images for MIC
        elif self.mask_mode in ['separatetrg', 'separatetrgaug']:
            masked_img = target_img
            masked_lbl = masked_plabel.unsqueeze(1)
            masked_seg_weight = masked_pweight
        else:
            raise NotImplementedError(self.mask_mode)

        # Apply color augmentation
        if 'aug' in self.mask_mode:
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': self.color_jitter_s,
                'color_jitter_p': self.color_jitter_p,
                'blur': random.uniform(0, 1),
                'mean': means[0].unsqueeze(0),
                'std': stds[0].unsqueeze(0)
            }
            masked_img, _ = strong_transform(strong_parameters, data=masked_img.clone())

        # Apply masking to image
        masked_img = self.mask_gen.mask_image(masked_img)

        # Train on masked images
        masked_loss = model.forward_train(
            masked_img,
            img_metas,
            masked_lbl,
            seg_weight=masked_seg_weight,
        )
        if self.mask_lambda != 1:
            masked_loss['decode.loss_seg'] *= self.mask_lambda

        # if self.debug:
        #     self.debug_output['Masked'] = model.debug_output
        #     if masked_seg_weight is not None:
        #         self.debug_output['Masked']['PL Weight'] = masked_seg_weight.cpu().numpy()

        return masked_loss

