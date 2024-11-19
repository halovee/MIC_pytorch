# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
import os
import platform
from torch import distributed as dist
from utils.filesio import FileClient


def runner_save_checkpoint(self,
                           out_dir,
                           filename_tmpl='iter_{}.pth',
                           meta=None,
                           save_optimizer=True,
                           create_symlink=True):
    """Save checkpoint to file.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta should be a dict or None, but got {type(meta)}')
    if self.meta is not None:
        meta.update(self.meta)
    meta.update(epoch=self.epoch + 1, iter=self.iter)
    filename = filename_tmpl.format(self.iter + 1)
    filepath = os.path.join(out_dir, filename)
    optimizer = self.optimizer if save_optimizer else None

    from ..hooks import ckp_save_checkpoint
    from utils import symlink
    ckp_save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
    if create_symlink:
        dst_file = os.path.join(out_dir, 'latest.pth')
        if platform.system() != 'Windows':
            symlink(filename, dst_file)
        else:
            import shutil
            shutil.copy(filepath, dst_file)

class CheckpointHook:
    """Save checkpoints periodically.
    """

    def __init__(self,
                 interval=-1,
                 by_epoch=True,
                 save_optimizer=True,
                 out_dir=None,
                 max_keep_ckpts=-1,
                 save_last=True,
                 sync_buffer=False,
                 **kwargs):
        # --------------- CheckpointHook -----------------
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.sync_buffer = sync_buffer

    def before_run(self, work_dir):
        if not self.out_dir:
            self.out_dir = work_dir

    def after_train_epoch(self, meta, iiter, epoch, max_epochs, model):
        if not self.by_epoch:
            return
        def every_n_epochs(n):
            return (epoch + 1) % n == 0 if n > 0 else False
        def is_last_epoch():
            return epoch + 1 == max_epochs
        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if every_n_epochs(self.interval) or (self.save_last and is_last_epoch):
            print(f'Saving checkpoint at {epoch + 1} epochs')
            if self.sync_buffer:
                allreduce_params(model.buffers())
            self._save_checkpoint(meta, epoch, iiter)

    def _save_checkpoint(self, meta, epoch, iiter):
        """Save the current checkpoint and delete unwanted checkpoint."""
        runner_save_checkpoint(self.out_dir, save_optimizer=self.save_optimizer, **self.args)
        if meta is not None:
            if self.by_epoch:
                cur_ckpt_filename = self.args.get('filename_tmpl', 'epoch_{}.pth').format(epoch + 1)
            else:
                cur_ckpt_filename = self.args.get('filename_tmpl', 'iter_{}.pth').format(iiter + 1)
            meta.setdefault('hook_msgs', dict())
            meta['hook_msgs']['last_ckpt'] = os.path.join(self.out_dir, cur_ckpt_filename)
        # remove other checkpoints
        if self.max_keep_ckpts > 0:
            if self.by_epoch:
                name = 'epoch_{}.pth'
                current_ckpt = epoch + 1
            else:
                name = 'iter_{}.pth'
                current_ckpt = iiter + 1
            redundant_ckpts = range(current_ckpt - self.max_keep_ckpts * self.interval, 0, -self.interval)
            filename_tmpl = self.args.get('filename_tmpl', name)
            for _step in redundant_ckpts:
                ckpt_path = os.path.join(self.out_dir, filename_tmpl.format(_step))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                else:
                    break

    def after_train_iter(self, meta, epoch, iiter, max_iters, model):
        if self.by_epoch:
            return
        def every_n_iters(n):
            return (iiter + 1) % n == 0 if n > 0 else False
        def is_last_iter():
            return iiter + 1 == max_iters
        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if every_n_iters(self.interval) or (self.save_last and is_last_iter()):
            print(f'Saving checkpoint at {iiter + 1} iterations')
            if self.sync_buffer:
                allreduce_params(model.buffers())
            self._save_checkpoint(meta, epoch, iiter)


def allreduce_params(params, coalesce=True, bucket_size_mb=-1):
    """Allreduce parameters.    """
    from utils import get_dist_info
    _, world_size = get_dist_info()
    if world_size == 1:
        return
    params = [param.data for param in params]
    if coalesce:
        _allreduce_coalesced(params, world_size, bucket_size_mb)
    else:
        for tensor in params:
            dist.all_reduce(tensor.div_(world_size))

def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    from torch._utils import _flatten_dense_tensors, _take_tensors, _unflatten_dense_tensors
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)

