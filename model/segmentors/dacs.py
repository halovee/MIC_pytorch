import math
import random
from copy import deepcopy
import numpy as np
import torch
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
import matplotlib.pyplot as plt
from configs import get_cfg_run, get_cfg_model
from model.basesegmentor_wrappers import auto_fp16
from model.base_module import BaseModule
from model.segmentors.base import BaseSegmentor
from model.segmentors.hrdaEncodeDecode import HRDAEncoderDecoder, crop
from model.uda.dacs_transforms import get_class_masks, strong_transform, get_mean_std
from model.uda.masking_consistency_module import MaskingConsistencyModule
from utils import downscale_label_ratio, add_prefix
from utils.losses.loss_utils import _parse_losses


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)
    return norm

class DACS(BaseSegmentor):

    def __init__(self, **kwargs):
        super(DACS, self).__init__()

        # UDADecorator中的构建
        self.model = HRDAEncoderDecoder(**kwargs)
        self.train_cfg = kwargs['train_cfg']
        self.test_cfg = kwargs['test_cfg']
        self.num_classes = kwargs['decode_head']['num_classes']

        # DACS中的构建
        cfg_uda = get_cfg_model('uda')
        max_iters = get_cfg_run('max_iters')
        cfg_uda.update({'max_iters': max_iters})
        self.local_iter = 0
        self.max_iters = max_iters
        self.source_only = cfg_uda['source_only']
        self.alpha = cfg_uda['alpha']
        self.pseudo_threshold = cfg_uda['pseudo_threshold']
        self.psweight_ignore_top = cfg_uda['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg_uda['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg_uda['imnet_feature_dist_lambda']
        self.fdist_classes = cfg_uda['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg_uda['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg_uda['mix']
        self.blur = cfg_uda['blur']
        self.color_jitter_s = cfg_uda['color_jitter_strength']
        self.color_jitter_p = cfg_uda['color_jitter_probability']
        self.mask_mode = cfg_uda['mask_mode']
        self.enable_masking = self.mask_mode is not None
        self.print_grad_magnitude = cfg_uda['print_grad_magnitude']
        assert self.mix == 'class'

        self.class_probs = {}
        if not self.source_only:
            self.ema_model = HRDAEncoderDecoder(**deepcopy(kwargs))
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg_uda)
        if self.enable_fdist:
            self.imnet_model = HRDAEncoderDecoder(**deepcopy(kwargs))
        else:
            self.imnet_model = None

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(self.model.parameters())
        mcp = list(self.ema_model.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.     """
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs


    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        if mask is not None:
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.model, HRDAEncoderDecoder) and self.model.feature_scale in self.model.feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.imnet_model.eval()
                feat_imnet = self.imnet_model.extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(gt_rescaled, HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(gt_rescaled, scale_factor, self.fdist_scale_min_ratio, self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s], fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.imnet_model.eval()
                feat_imnet = self.imnet_model.extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor, self.fdist_scale_min_ratio, self.num_classes, 255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay], fdist_mask)
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses({'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=logits.device)

        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def check_nan(self, img, img_name=""):
        contains_nan = torch.isnan(img).any().item()
        nan_positions = torch.isnan(img)
        print(f'Contains NaN: {f"{img_name}_Yes" if contains_nan else f"{img_name}_No"}')
        if contains_nan:
            print(f'Number of NaNs in {img_name}: {torch.sum(nan_positions).item()}')

    def check_for_nans(self, model):
        for name, param in model.state_dict().items():
            if torch.isnan(param).any().item():
                print(f"参数 {name} 包含 nan 值")
                return True
        return False

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      target_img,
                      target_img_metas,
                      rare_class=None,
                      valid_pseudo_mask=None):
        """Forward function for training. """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # for i in [img, target_img, gt_semantic_seg, valid_pseudo_mask]:
        #     self.check_nan(i)

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
        if self.local_iter > 0:
            self._update_ema(self.local_iter)
        if self.mic is not None:
            self.mic.update_weights(self.model, self.local_iter)

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        # self.check_nan(means, 'means')
        # self.check_nan(stds, 'stds')

        # Train on source images
        clean_losses = self.model.forward_train( img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        for i in range(len(src_feat)):
            self.check_nan(src_feat[i])
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if not self.check_for_nans(self.model):
            print('No NaNs in model parameters')
        if self.print_grad_magnitude:
            params = self.model.backbone.parameters()
            seg_grads = [p.grad.detach().clone() for p in params if p.grad is not None]
            grad_mag = calc_grad_magnitude(seg_grads)
            print(f'Seg. Grad.: {grad_mag}')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg, src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()
            if not self.check_for_nans(self.imnet_model):
                print('No NaNs in imnet_model parameters')
            if self.print_grad_magnitude:
                params = self.model.backbone.parameters()
                fd_grads = [p.grad.detach() for p in params if p.grad is not None]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                print(f'Fdist Grad.: {grad_mag}')
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        pseudo_label, pseudo_weight = None, None
        if not self.source_only:
            # Generate pseudo-label
            for m in self.ema_model.modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            ema_logits = self.ema_model.generate_pseudo_label(target_img, target_img_metas)
            pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(ema_logits)
            del ema_logits

            pseudo_weight = self.filter_valid_pseudo_region(pseudo_weight, valid_pseudo_mask)
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)
            if not self.check_for_nans(self.ema_model):
                print('No NaNs in ema_model parameters')

            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mixed_seg_weight = pseudo_weight.clone()
            mix_masks = get_class_masks(gt_semantic_seg)
            # 检查 mix_masks[i] 是否包含 NaN 值或异常值 (例如，所有值为 0 或 1)。重点关注 mix_masks[1]。
            # 特别是检查当 mix_masks[i] 中的某些类别缺失时，混合操作的行为。
            self.check_nan(mix_masks[1], 'mix_masks[1]')
            self.check_nan(img[1], 'img[1]')
            self.check_nan(target_img[1], 'target_img[1]')

            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                self.check_nan(img[i], f'img_{i}')
                self.check_nan(target_img[i], f'target_img_{i}')
                self.check_nan(gt_semantic_seg[i][0], f'gt_semantic_seg_{i}[0]')
                self.check_nan(pseudo_label[i], f'pseudo_label_{i}')
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((img[i], target_img[i])),
                    target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
                _, mixed_seg_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
                self.check_nan(mixed_lbl[i], f'mixed_lbl_{i}')
                self.check_nan(mixed_img[i], f'mixed_img_{i}')
                self.check_nan(mixed_seg_weight[i], f'mixed_seg_weight_{i}')
            del gt_pixel_weight
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)
            self.check_nan(mixed_img, 'mixed_img')
            self.check_nan(mixed_lbl, 'mixed_lbl')
            self.check_nan(mixed_seg_weight, 'mixed_seg_weight')

            # Train on mixed images
            mix_losses = self.model.forward_train( mixed_img, img_metas, mixed_lbl, seg_weight=mixed_seg_weight, return_feat=False )
            if not self.check_for_nans(self.model):
                print('No NaNs in model parameters')
            # seg_debug['Mix'] = self.model.debug_output
            mix_losses = add_prefix(mix_losses, 'mix')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            mix_loss.backward()
            if not self.check_for_nans(self.model):
                print('No NaNs in model parameters')

        # Masked Training
        if self.enable_masking and self.mask_mode.startswith('separate'):
            masked_loss = self.mic(self.model, img, img_metas, gt_semantic_seg, target_img, target_img_metas, valid_pseudo_mask, pseudo_label, pseudo_weight)
            if not self.check_for_nans(self.mic):
                print('No NaNs in mic parameters')
            # seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, 'masked')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()

        self.local_iter += 1

        return log_vars

    # ---------------------UDADecorator --------------------------------
    def extract_feat(self, img):
        """Extract features from images."""
        return self.model.extract_feat(img)

    def encode_decode(self, img, img_metas, upscale_pred=True):
        """Encode images with backbone and decode into a semantic segmentation map of the same size as input."""
        return self.model.encode_decode(img, img_metas, upscale_pred)

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.
        """
        return self.model.inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        return self.model.simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.
        """
        return self.model.aug_test(imgs, img_metas, rescale)



