import json

import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

from data.convert_datasets import save_class_stats
from data.datasets.custom import CustomDataset
from data.pipelines.aug_pipeline import Compose
from data.datasets.cityscapes import CityscapesDataset
import os.path as osp
from utils import mkdir_or_exist, scandir, track_parallel_progress, track_progress


def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    # PIL does not work with the image format, but cv2 does
    label = cv2.imread(file, cv2.IMREAD_UNCHANGED)[:, :, -1]
    # mapping based on README.txt from SYNTHIA_RAND_CITYSCAPES
    id_to_trainid = {
        3: 0,
        4: 1,
        2: 2,
        21: 3,
        5: 4,
        7: 5,
        15: 6,
        9: 7,
        6: 8,
        16: 9,  # not present in synthia
        1: 10,
        10: 11,
        17: 12,
        8: 13,
        18: 14,  # not present in synthia
        19: 15,
        20: 16,  # not present in synthia
        12: 17,
        11: 18
    }
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    for k, v in id_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    new_file = file.replace('.png', '_labelTrainIds.png')
    assert file != new_file
    sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats


class SynthiaDataset(CustomDataset):
    CLASSES = CityscapesDataset.CLASSES
    PALETTE = CityscapesDataset.PALETTE

    def __init__(self,
                 id_out_dir='./data/ids2path',
                 **kwargs):
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')
        super(SynthiaDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelTrainIds.png',
            split=None,
            **kwargs)
        self.id_out_dir = id_out_dir
        self.convert_id = self.build_convert_id(out_dir=id_out_dir,nproc=8)

    def build_convert_id(self, out_dir=None, nproc=4):
        import os
        dataset_name = self.__class__.__name__
        out_dir = out_dir if out_dir else self.data_root
        self.sample_class_stats_dir = osp.join(out_dir, f'{dataset_name}_sample_class_stats.json')
        if not os.path.exists(self.sample_class_stats_dir):
            mkdir_or_exist(out_dir)
            gt_dir = osp.join(self.data_root, self.ann_dir)
            poly_files = []
            for poly in scandir(gt_dir, suffix=tuple(f'{i}.png' for i in range(10)), recursive=True):
                poly_file = osp.join(gt_dir, poly)
                poly_files.append(poly_file)
            poly_files = sorted(poly_files)
            only_postprocessing = False
            if not only_postprocessing:
                if nproc > 1:
                    sample_class_stats = track_parallel_progress(convert_to_train_id, poly_files, nproc)
                else:
                    sample_class_stats = track_progress(convert_to_train_id, poly_files)
            else:
                with open(self.sample_class_stats_dir, 'r') as of:
                    sample_class_stats = json.load(of)

        save_class_stats(out_dir, sample_class_stats, dataset_name)

