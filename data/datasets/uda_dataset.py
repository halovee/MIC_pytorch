

import json
import os.path as osp

import numpy as np
import torch
from data.datasets.cityscapes import CityscapesDataset
from data.pipelines import DataContainer
from utils.logger import print_log


def get_rcs_class_probs(data_root, temperature):
    '''计算数据集的稀有类别概率 (Rare Class Probabilities)'''
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)   # 读取样本类别统计信息 :  将读取的 JSON 文件内容解析成 Python 字典，并存储在 sample_class_stats 变量中。
    overall_class_stats = {}                 # 统计所有类别的像素数
    for s in sample_class_stats:             # 遍历样本类别统计信息
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {k: v for k, v in sorted(overall_class_stats.items(), key=lambda item: item[1])}
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


def get_crop_bbox(img_size, crop_size):
    """Randomly get a crop bounding box."""
    assert len(img_size) == len(crop_size)
    assert len(img_size) == 2
    margin_h = max(img_size[0] - crop_size[0], 0)
    margin_w = max(img_size[1] - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2


class UDADataset(object):

    def __init__(self, source, target, cfg):
        self.source = source
        self.target = target
        self.ignore_index = target.ignore_index
        self.CLASSES = target.CLASSES
        self.PALETTE = target.PALETTE
        assert target.ignore_index == source.ignore_index
        assert target.CLASSES == source.CLASSES
        assert target.PALETTE == source.PALETTE

        self.sync_crop_size = cfg.get('sync_crop_size')     # 用于控制图像裁剪的大小和同步性。确保图像尺寸统一和多图像裁剪对应关系。
        rcs_cfg = cfg.get('rare_class_sampling')            # 是在不平衡数据集上的训练中常用的采样技术，提高模型对稀有类别的识别能力。

        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            source_class_name = source.__class__.__name__
            cfg_source = cfg['source_gta'] if source_class_name == 'GTADataset' else cfg['source_syn']
            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(cfg_source['data_root'], self.rcs_class_temp)
            print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

            with open(osp.join(cfg_source['data_root'], 'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {int(k): v for k, v in samples_with_class_and_n.items() if int(k) in self.rcs_classes}
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.source.img_infos):
                file = dic['ann']['seg_map']
                if isinstance(self.source, CityscapesDataset):
                    file = file.split('/')[-1]
                self.file_to_idx[file] = i

    def synchronized_crop(self, s1, s2):
        if self.sync_crop_size is None:
            return s1, s2
        orig_crop_size = s1['img'].data.shape[1:]
        crop_y1, crop_y2, crop_x1, crop_x2 = get_crop_bbox(orig_crop_size, self.sync_crop_size)
        for i, s in enumerate([s1, s2]):
            for key in ['img', 'gt_semantic_seg', 'valid_pseudo_mask']:
                if key not in s:
                    continue
                s[key] = DataContainer(s[key].data[:, crop_y1:crop_y2, crop_x1:crop_x2], stack=s[key]._stack)
        return s1, s2

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['gt_semantic_seg'].data == c)
                # print(f'{j}: {n_class}')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                s1 = self.source[i1]
        i2 = np.random.choice(range(len(self.target)))
        s2 = self.target[i2]
        s1, s2 = self.synchronized_crop(s1, s2)
        out = {**s1, 'target_img_metas': s2['img_metas'], 'target_img': s2['img']}
        if 'valid_pseudo_mask' in s2:
            out['valid_pseudo_mask'] = s2['valid_pseudo_mask']
        return out

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            s1 = self.source[idx // len(self.target)]
            s2 = self.target[idx % len(self.target)]
            s1, s2 = self.synchronized_crop(s1, s2)
            out = {**s1, 'target_img_metas': s2['img_metas'], 'target_img': s2['img']}
            if 'valid_pseudo_mask' in s2:
                out['valid_pseudo_mask'] = s2['valid_pseudo_mask']
            return out

    def __len__(self):
        return len(self.source) * len(self.target)
