import os
import platform
import warnings
import pickle
import shutil
import tempfile
import time

import torch
from math import inf
from torch.utils.data import DataLoader

from model.runner.hooks import runner_save_checkpoint
from utils import mkdir_or_exist, ProgressBar, get_dist_info
from utils.filesio import dump, load
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm

def single_gpu_test(model, data_loader):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.extend(result)

        # Assume result has the same length of batch_size
        # refer to https://github.com/open-mmlab/mmcv/issues/985
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, 
                   data_loader, 
                   tmpdir=None, 
                   gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting
    ``gpu_collect=True``, it encodes results to gpu tensors and use gpu
    communication for results collection. On cpu mode it saves the results on
    different gpus to ``tmpdir`` and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            batch_size_all = batch_size * world_size
            if batch_size_all + prog_bar.completed > len(dataset):
                batch_size_all = len(dataset) - prog_bar.completed
            for _ in range(batch_size_all):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    dump(result_part, os.path.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, f'part_{i}.pkl')
            part_result = load(part_file)
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list): Result list containing result parts
            to be collected.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list: The collected results.
    """
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            # When data is severely insufficient, an empty part_result
            # on a certain gpu could makes the overall outputs empty.
            if part_result:
                part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


class EvalHook(object):
    """Non-Distributed evaluation hook.   """
    rule_map = {'greater': lambda x, y: x > y, 'less': lambda x, y: x < y}
    init_value_map = {'greater': -inf, 'less': inf}
    greater_keys = [
        'acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'mDice', 'mIoU',
        'mAcc', 'aAcc'
    ]
    less_keys = ['loss']

    def __init__(self,
                 # ------------------eval_hooks.py中的参数
                 dataloader,
                 *args,
                 by_epoch=False,
                 efficient_test=False,
                 # ------------------evaluaion.py中的参数
                 start=None,
                 interval=1,
                 save_best=None,
                 rule=None,
                 **eval_kwargs):
        # ----------------- eval_hooks.py -----------------
        self.efficient_test = efficient_test
        # ----------------- evaluation.py -----------------
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f'dataloader must be a pytorch DataLoader, but got {type(dataloader)}')
        if interval <= 0:
            raise ValueError(f'interval must be a positive number, but got {interval}')

        assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean'

        if start is not None and start < 0:
            raise ValueError(f'The evaluation start epoch {start} is smaller '
                             f'than 0')
        # ----------------- EvalHook -----------------
        self.dataloader = dataloader
        self.interval = interval
        self.start = start
        self.by_epoch = by_epoch

        assert isinstance(save_best, str) or save_best is None, '""save_best"" should be a str or None 'f'rather than {type(save_best)}'
        self.save_best = save_best
        self.eval_kwargs = eval_kwargs
        self.initial_flag = True

        if self.save_best is not None:
            self.best_ckpt_path = None
            self._init_rule(rule, self.save_best)


    def _init_rule(self, rule, key_indicator):
        """初始化规则、key_indicator、comparison_func 和最佳分数。

        以下是确定哪个规则用于关键指示器的规则
        当规则不特定时（请注意，键指示符匹配不区分大小写）：
        1. 如果键指示符在 ''self.greater_keys'' 中，则规则将被指定为 'greater'。
        2. 或者，如果关键指示符在 ''self.less_keys'' 中，则规则将被指定为 'less'。
        3. 或者，如果键指示符等于 ''self.greater_keys'' 中任何一项的子字符串，则规则将被指定为 'greater'。
        4. 或者，如果键指示符等于 ''self.less_keys'' 中任何一项中的子字符串，则规则将被指定为 'less'。

        参数：
            rule （str |None）：最佳分数的比较规则。
            key_indicator （str |None）：用于确定比较规则的关键指示符。
        """
        if rule not in self.rule_map and rule is not None:
            raise KeyError(f'rule must be greater, less or None, but got {rule}.')

        if rule is None:
            if key_indicator != 'auto':
                # `_lc` here means we use the lower case of keys for case-insensitive matching
                key_indicator_lc = key_indicator.lower()
                greater_keys = [key.lower() for key in self.greater_keys]
                less_keys = [key.lower() for key in self.less_keys]

                if key_indicator_lc in greater_keys:
                    rule = 'greater'
                elif key_indicator_lc in less_keys:
                    rule = 'less'
                elif any(key in key_indicator_lc for key in greater_keys):
                    rule = 'greater'
                elif any(key in key_indicator_lc for key in less_keys):
                    rule = 'less'
                else:
                    raise ValueError(f'Cannot infer the rule for key {key_indicator}, thus a specific rule must be specified.')
        self.rule = rule
        self.key_indicator = key_indicator
        if self.rule is not None:
            self.compare_func = self.rule_map[self.rule]

    def before_run(self, meta):
        if self.save_best is not None:
            if meta is None:
                warnings.warn('runner.meta is None. Creating an empty one.')
                meta = dict()
            meta.setdefault('hook_msgs', dict())
            self.best_ckpt_path = meta['hook_msgs'].get('best_ckpt', None)

    def before_train_iter(self, model, log_buffer, epoch, iiter, logger, work_dir, meta):
        """Evaluate the model only at the start of training by iteration."""
        if self.by_epoch or not self.initial_flag:
            return
        if self.start is not None and iiter >= self.start:
            self.after_train_iter(model, log_buffer, epoch, iiter, logger, work_dir, meta)
        self.initial_flag = False

    def before_train_epoch(self, model, log_buffer, epoch, iiter, logger, work_dir, meta):
        """Evaluate the model only at the start of training by epoch."""
        if not (self.by_epoch and self.initial_flag):
            return
        if self.start is not None and epoch >= self.start:
            self.after_train_epoch(model, log_buffer, epoch, iiter, logger, work_dir, meta)
        self.initial_flag = False

    def after_train_iter(self, model, log_buffer, epoch, iiter, logger, work_dir, meta):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch:
            self._do_evaluate(model, log_buffer, epoch, iiter, logger, work_dir, meta)

    def after_train_epoch(self, model, log_buffer, epoch, iiter, logger, work_dir, meta):
        """Called after every training epoch to evaluate the results."""
        # if self.by_epoch and self._should_evaluate():
        if not self.by_epoch:
            self._do_evaluate(model, log_buffer, epoch, iiter, logger, work_dir, meta)

    def _do_evaluate(self, model, log_buffer, epoch, iiter, logger, work_dir, meta):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(epoch, iiter):
            return
        results = single_gpu_test(model, self.dataloader)
        log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(results, logger, log_buffer)
        if self.save_best:
            self._save_ckpt(key_score, epoch, iiter, work_dir, meta, logger)

    def _should_evaluate(self, epoch, iiter):
        """判断是否进行评估。

        以下是判断是否进行评估的规则：
        1. 它不会在 epoch/iteration interval 内执行评估，该间隔由 ''self.interval'' 决定。
        2. 如果开始时间大于当前时间，则不会进行评估。
        3. 当当前时间大于开始时间时，但在 epoch/iteration interval 内不会执行评估。

        Returns:
            bool: 指示是否执行评估的标志。
        """
        def every_n_iters(n):
            return (iiter + 1) % n == 0 if n > 0 else False
        def every_n_epochs(n):
            return (epoch + 1) % n == 0 if n > 0 else False
        if self.by_epoch:
            current = epoch
            check_time = every_n_epochs
        else:
            current = iiter
            check_time = every_n_iters

        if self.start is None:
            if not check_time(self.interval):
                # No evaluation during the interval.
                return False
        elif (current + 1) < self.start:
            # No evaluation if start is larger than the current time.
            return False
        else:
            # Evaluation only at epochs/iters 3, 5, 7... if start==3 and interval==2
            if (current + 1 - self.start) % self.interval:
                return False
        return True

    def _save_ckpt(self, key_score, epoch, iiter, work_dir, meta, logger):
        """Save the best checkpoint.

        It will compare the score according to the compare function, write
        related information (best score, best checkpoint path) and save the
        best checkpoint into ``work_dir``.
        """
        if self.by_epoch:
            current = f'epoch_{epoch + 1}'
            cur_type, cur_time = 'epoch', epoch + 1
        else:
            current = f'iter_{iiter + 1}'
            cur_type, cur_time = 'iter', iiter + 1

        best_score = meta['hook_msgs'].get(
            'best_score', self.init_value_map[self.rule])
        if self.compare_func(key_score, best_score):
            best_score = key_score
            meta['hook_msgs']['best_score'] = best_score

            if self.best_ckpt_path and os.path.isfile(self.best_ckpt_path):
                os.remove(self.best_ckpt_path)

            best_ckpt_name = f'best_{self.key_indicator}_{current}.pth'
            self.best_ckpt_path = os.path.join(work_dir, best_ckpt_name)
            meta['hook_msgs']['best_ckpt'] = self.best_ckpt_path

            runner_save_checkpoint(work_dir, best_ckpt_name, create_symlink=False)
            logger.info(f'Now best checkpoint is saved as {best_ckpt_name}.')
            logger.info(f'Best {self.key_indicator} is {best_score:0.4f} at {cur_time} {cur_type}.')

    def evaluate(self, results, logger, log_buffer):
        """Evaluate the results.

        Args:
            runner (:obj:`mmcv.Runner`): The underlined training runner.
            results (list): Output results.
        """
        eval_res = self.dataloader.dataset.evaluate(results, logger=logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            log_buffer.output[name] = val
        log_buffer.ready = True

        if self.save_best is not None:
            if self.key_indicator == 'auto':
                # infer from eval_results
                self._init_rule(self.rule, list(eval_res.keys())[0])
            return eval_res[self.key_indicator]

        return None




class DistEvalHook(EvalHook):
    """Distributed evaluation hook.  """

    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1,
                 by_epoch=True,
                 save_best=None,
                 rule=None,
                 broadcast_bn_buffer=True,
                 tmpdir=None,
                 gpu_collect=False,

                 efficient_test=False,
                 **eval_kwargs):
                 super().__init__(
                     dataloader,
                     start=start,
                     interval=interval,
                     by_epoch=by_epoch,
                     save_best=save_best,
                     rule=rule,
                     **eval_kwargs)
                 self.broadcast_bn_buffer = broadcast_bn_buffer
                 self.tmpdir = tmpdir
                 self.gpu_collect = gpu_collect
                 self.efficient_test = efficient_test


    def _do_evaluate(self, runner_model, log_buffer, epoch, iiter,  logger, work_dir, meta):
        """perform evaluation and save ckpt."""

        if self.broadcast_bn_buffer:
            model = runner_model
            for name, module in model.named_modules():
                if isinstance(module, _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(epoch, iiter):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = os.path.join(work_dir, '.eval_hook')

        results = multi_gpu_test(
            runner_model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        rank, _ = get_dist_info()
        if rank == 0:
            print('\n')
            log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(results, logger, log_buffer)

            if self.save_best:
                self._save_ckpt(key_score, epoch, iiter, work_dir, meta, logger)

