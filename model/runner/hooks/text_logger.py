# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os
from collections import OrderedDict

import torch
import torch.distributed as dist

from configs.gtaHR2csHR_mic_hrda_s2_8e10a import runner
from utils import FileClient, dump, scandir, get_dist_info, get_time_str
from utils.py_utils import is_tuple_of


class TextLoggerHook(object):
    """Logger hook in text.
    """

    def __init__(self,
                 by_epoch=True,
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 interval_exp_name=1000,
                 **kwargs):
        super(TextLoggerHook, self).__init__()
        # --------------- TextLoggerHook -----------------
        self.interval = interval
        self.ignore_last = ignore_last
        self.reset_flag = reset_flag

        self.by_epoch = by_epoch
        self.time_sec_tot = 0
        self.interval_exp_name = interval_exp_name

    def before_run(self, meta, work_dir, timestamp, iiter):
        self.reset_flag = True
        self.start_iter = iiter
        self.json_log_path = os.path.join(work_dir, f'{timestamp}.log.json')
        if meta is not None:
            self._dump_log(meta)

    def _get_max_memory(self, model):
        device = getattr(model, 'output_device', None)
        mem = torch.cuda.max_memory_allocated(device=device)
        mem_mb = torch.tensor([mem / (1024 * 1024)], dtype=torch.int, device=device)
        _, world_size = get_dist_info()
        if world_size > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
        return mem_mb.item()

    def _log_info(self, log_dict, meta, iiter, inner_iter, logger, max_iters, data_loader):
        # print exp name for users to distinguish experiments at every ``interval_exp_name`` iterations and the end of each epoch
        def every_n_iters(n):
            return (iiter + 1) % n == 0 if n > 0 else False
        def end_of_epoch():
            return inner_iter + 1 == len(data_loader)
        if meta is not None and 'exp_name' in meta:
            if (every_n_iters(self.interval_exp_name)) or (self.by_epoch and end_of_epoch()):
                exp_info = f'Exp name: {meta["exp_name"]}'
                logger.info(exp_info)

        if log_dict['mode'] == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.3e}')
                lr_str = ' '.join(lr_str)
            else:
                lr_str = f'lr: {log_dict["lr"]:.3e}'

            # by epoch: Epoch [4][100/1000]
            # by iter:  Iter [100/100000]
            if self.by_epoch:
                log_str = f'Epoch [{log_dict["epoch"]}][{log_dict["iter"]}/{len(data_loader)}]\t'
            else:
                log_str = f'Iter [{log_dict["iter"]}/{max_iters}]\t'
            log_str += f'{lr_str}, '

            if 'time' in log_dict.keys():
                self.time_sec_tot += (log_dict['time'] * self.interval)
                time_sec_avg = self.time_sec_tot / (iiter - self.start_iter + 1)
                eta_sec = time_sec_avg * (max_iters - iiter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                log_str += f'eta: {eta_str}, '
                log_str += f'time: {log_dict["time"]:.3f}, data_time: {log_dict["data_time"]:.3f}, '
                # statistic memory
                if torch.cuda.is_available():
                    log_str += f'memory: {log_dict["memory"]}, '
        else:
            # val/test time
            # here 1000 is the length of the val dataloader
            # by epoch: Epoch[val] [4][1000]
            # by iter: Iter[val] [1000]
            if self.by_epoch:
                log_str = f'Epoch({log_dict["mode"]}) [{log_dict["epoch"]}][{log_dict["iter"]}]\t'
            else:
                log_str = f'Iter({log_dict["mode"]}) [{log_dict["iter"]}]\t'

        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in [
                'mode', 'Epoch', 'iter', 'lr', 'time', 'data_time',
                'memory', 'epoch'
            ]:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)

        logger.info(log_str)

    def _dump_log(self, log_dict):
        # dump log in json format
        json_log = OrderedDict()
        for k, v in log_dict.items():
            json_log[k] = self._round_float(v)
        # only append log at last line
        rank , _ = get_dist_info()
        if rank == 0:
            with open(self.json_log_path, 'a+') as f:
                dump(json_log, f, file_format='json')
                f.write('\n')

    def _round_float(self, items):
        if isinstance(items, list):
            return [self._round_float(item) for item in items]
        elif isinstance(items, float):
            return round(items, 5)
        else:
            return items

    # ----------------------- runner -----------------------
    def current_lr(self , optimizer):
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        if isinstance(optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in optimizer.param_groups]
        elif isinstance(optimizer, dict):
            lr = dict()
            for name, optim in optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError('lr is not applicable because optimizer does not exist.')
        return lr
    
    def log(self,
            log_buffer,
            optimizer,
            meta,
            _mode,
            iiter,
            _epoch,
            inner_iter,
            logger,
            max_iters,
            data_loader,
            model):
        if 'eval_iter_num' in log_buffer.output:
            # this doesn't modify runner.iter and is regardless of by_epoch
            cur_iter = log_buffer.output.pop('eval_iter_num')
        else:
            cur_iter = self.get_iter(iiter, inner_iter=True)

        log_dict = OrderedDict(
            mode=self.get_mode(_mode, log_buffer),
            epoch=self.get_epoch(_mode, _epoch),
            iter=cur_iter)

        # only record lr of the first param group
        cur_lr = self.current_lr(optimizer)
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        if 'time' in log_buffer.output:
            # statistic memory
            if torch.cuda.is_available():
                log_dict['memory'] = self._get_max_memory(model)

        log_dict = dict(log_dict, **log_buffer.output)

        self._log_info(log_dict, meta, iiter, inner_iter, logger, max_iters, data_loader)
        self._dump_log(log_dict)
        return log_dict

    def before_epoch(self, log_buffer):
        log_buffer.clear()

# ------------------------ LoggerHook ------------------------
    def after_train_iter(self,
                         log_buffer,
                         optimizer,
                         meta,
                         mode,
                         iiter,
                         epoch,
                         inner_iter,
                         logger,
                         max_iters,
                         data_loader,
                         model):
        def every_n_inner_iters(n):
            return (inner_iter + 1) % n == 0 if n > 0 else False
        def every_n_iters(n):
            return (iiter + 1) % n == 0 if n > 0 else False
        def end_of_epoch():
            return inner_iter + 1 == len(data_loader)

        if self.by_epoch and every_n_inner_iters(self.interval):
            log_buffer.average(self.interval)
        elif not self.by_epoch and every_n_iters(self.interval):
            log_buffer.average(self.interval)
        elif end_of_epoch() and not self.ignore_last:
            # not precise but more stable
            log_buffer.average(self.interval)

        if log_buffer.ready:
            self.log(log_buffer, optimizer, meta, mode, iiter, epoch, inner_iter, logger, max_iters, data_loader, model)
            if self.reset_flag:
                log_buffer.clear_output()

    def after_train_epoch(self,
                          log_buffer,
                          optimizer,
                          meta,
                          mode,
                          iiter,
                          epoch,
                          inner_iter,
                          logger,
                          max_iters,
                          data_loader,
                          model):
        if log_buffer.ready:
            self.log(log_buffer, optimizer, meta, mode, iiter, epoch, inner_iter, logger, max_iters, data_loader, model)
            if self.reset_flag:
                log_buffer.clear_output()

    def after_val_epoch(self,
                        log_buffer,
                        optimizer,
                        meta,
                        mode,
                        iiter,
                        epoch,
                        inner_iter,
                        logger,
                        max_iters,
                        data_loader,
                        model):
        log_buffer.average()
        self.log(log_buffer, optimizer, meta, mode, iiter, epoch, inner_iter, logger, max_iters, data_loader, model)
        if self.reset_flag:
            log_buffer.clear_output()

    def get_iter(self, iiter, inner_iter=False):
        """Get the current training iteration step."""
        if self.by_epoch and inner_iter:
            current_iter = inner_iter + 1
        else:
            current_iter = iiter + 1
        return current_iter

    def get_mode(self, _mode, log_buffer):
        if _mode == 'train':
            if 'time' in log_buffer.output:
                mode = 'train'
            else:
                mode = 'val'
        elif _mode == 'val':
            mode = 'val'
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', but got {_mode}")
        return mode

    def get_epoch(self, _mode, _epoch):
        if _mode == 'train':
            epoch = _epoch + 1
        elif _mode == 'val':
            # normal val mode
            # runner.epoch += 1 has been done before val workflow
            epoch = _epoch
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', but got {_mode}")
        return epoch