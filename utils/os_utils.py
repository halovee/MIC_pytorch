import os.path as osp
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable
import torch
import cv2
from py3nvml.py3nvml import *

def prt_gpu(stat = ''):
    # 初始化
    nvmlInit()
    # 获取第一个GPU的显存信息
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"总显存: {info.total / (1024 ** 3)} GiB")
    print(f"已使用显存: {info.used / (1024 ** 3)} GiB")
    print(f"空闲显存: {info.free / (1024 ** 3)} GiB")
    # 关闭nvml
    nvmlShutdown()
    print(stat)

def prt_n(stat = ''):
    import psutil
    # 获取当前进程
    current_process = psutil.Process(os.getpid())
    # 获取当前进程占用的内存信息（以字节为单位）
    memory_info = current_process.memory_info()
    memory_usage = memory_info.rss  # rss 表示常驻内存集大小
    # 获取当前系统中的进程总数
    process_count = len(psutil.pids())
    print(f"当前进程ID: {os.getpid()}")
    print(f"当前进程占用的内存: {memory_usage / (1024 ** 3)} GiB")
    print(f"系统中的进程总数: {process_count}")
    print(stat)

def get_dist_info():
    from torch import distributed as dist
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def get_time_str():
    import time
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def prt(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of_{model.__class__.__name__}_ trainable parameters: {num_parameters}")

def add_prefix(inputs, prefix):
    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def env_info():
    env_info_dict = collect_env()  # log env info
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '\n' + '-' * 60 + '\n'
    env_info = dash_line + 'Environment info:\n' + env_info + '\n' + dash_line
    # print('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    # print(f'Distributed training: {distributed}' + '\n' + dash_line)  # log some basic info
    return env_info

def collect_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - CUDA available: Bool, indicating if CUDA is available.
            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.
    """
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

        from functools import partial
        TORCH_VERSION = torch.__version__
        is_rocm = False
        if TORCH_VERSION != 'parrots':
            try:
                from torch.utils.cpp_extension import ROCM_HOME
                is_rocm = True if ((torch.version.hip is not None) and
                                   (ROCM_HOME is not None)) else False
            except ImportError:
                pass
        if is_rocm:
            from torch.utils.cpp_extension import ROCM_HOME
            CUDA_HOME = ROCM_HOME
        else:
            from torch.utils.cpp_extension import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME
        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    f'"{nvcc}" -V | tail -n1', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

    try:
        gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
        gcc = gcc.decode('utf-8').strip()
        env_info['GCC'] = gcc
    except subprocess.CalledProcessError:  # gcc is unavailable
        env_info['GCC'] = 'n/a'

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()

    try:
        import torchvision
        env_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    env_info['OpenCV'] = cv2.__version__

    return env_info


def scandir(dir_path, suffix=None, recursive=False, case_sensitive=True):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | :obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the directory. Default: False.
        case_sensitive (bool, optional) : If set to False, ignore the case of suffix. Default: True.

    Returns:
        A generator for all the interested files with relative paths.
    """
    from pathlib import Path
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    if suffix is not None and not case_sensitive:
        suffix = suffix.lower() if isinstance(suffix, str) else tuple(
            item.lower() for item in suffix)

    root = dir_path

    def _scandir(dir_path, suffix, recursive, case_sensitive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                _rel_path = rel_path if case_sensitive else rel_path.lower()
                if suffix is None or _rel_path.endswith(suffix):
                    yield rel_path
            elif recursive and os.path.isdir(entry.path):
                # scan recursively if entry.path is a directory
                yield from _scandir(entry.path, suffix, recursive,
                                    case_sensitive)

    return _scandir(dir_path, suffix, recursive, case_sensitive)

from time import time
class Timer:
    """A flexible Timer class.    """

    def __init__(self, start=True, print_tmpl=None):
        self._is_running = False
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        if start:
            self.start()

    @property
    def is_running(self):
        """bool: indicate whether the timer is running"""
        return self._is_running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        print(self.print_tmpl.format(self.since_last_check()))
        self._is_running = False

    def start(self):
        """Start the timer."""
        if not self._is_running:
            self._t_start = time()
            self._is_running = True
        self._t_last = time()

    def since_start(self):
        """Total time since the timer is started.

        Returns:
            float: Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        self._t_last = time()
        return self._t_last - self._t_start

    def since_last_check(self):
        """Time since the last checking.

        Either :func:`since_start` or :func:`since_last_check` is a checking
        operation.

        Returns:
            float: Time in seconds.
        """
        if not self._is_running:
            raise TimerError('timer is not running')
        dur = time() - self._t_last
        self._t_last = time()
        return dur


class TimerError(Exception):

    def __init__(self, message):
        self.message = message
        super(TimerError, self).__init__(message)

_g_timers = {}  # global timers


class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = os.get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(f'[{" " * self.bar_width}] 0/{self.task_num}, '
                            'elapsed: 0s, ETA:')
        else:
            self.file.write('completed: 0, elapsed: 0s')
        self.file.flush()
        self.timer = Timer()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = f'\r[{{}}] {self.completed}/{self.task_num}, ' \
                  f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
                  f'ETA: {eta:5}s'

            bar_width = min(self.bar_width,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} tasks/s')
        self.file.flush()

def init_pool(process_num, initializer=None, initargs=None):
    from multiprocessing import Pool
    if initializer is None:
        return Pool(process_num)
    elif initargs is None:
        return Pool(process_num, initializer)
    else:
        if not isinstance(initargs, tuple):
            raise TypeError('"initargs" must be a tuple')
        return Pool(process_num, initializer, initargs)

def track_parallel_progress(func,
                            tasks,
                            nproc,
                            initializer=None,
                            initargs=None,
                            bar_width=50,
                            chunksize=1,
                            skip_first=False,
                            keep_order=True,
                            file=sys.stdout):
    """Track the progress of parallel task execution with a progress bar.

    The built-in :mod:`multiprocessing` module is used for process pools and
    tasks are done with :func:`Pool.map` or :func:`Pool.imap_unordered`.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        nproc (int): Process (worker) number.
        initializer (None or callable): Refer to :class:`multiprocessing.Pool`
            for details.
        initargs (None or tuple): Refer to :class:`multiprocessing.Pool` for
            details.
        chunksize (int): Refer to :class:`multiprocessing.Pool` for details.
        bar_width (int): Width of progress bar.
        skip_first (bool): Whether to skip the first sample for each worker
            when estimating fps, since the initialization step may takes
            longer.
        keep_order (bool): If True, :func:`Pool.imap` is used, otherwise
            :func:`Pool.imap_unordered` is used.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start, file=file)
    results = []
    if keep_order:
        gen = pool.imap(func, tasks, chunksize)
    else:
        gen = pool.imap_unordered(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    prog_bar.file.write('\n')
    pool.close()
    pool.join()
    return results

def track_progress(func, tasks, bar_width=50, file=sys.stdout, **kwargs):
    """Track the progress of tasks execution with a progress bar.

    Tasks are done with a simple for-loop.

    Args:
        func (callable): The function to be applied to each task.
        tasks (list or tuple[Iterable, int]): A list of tasks or
            (tasks, total num).
        bar_width (int): Width of progress bar.

    Returns:
        list: The task results.
    """
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    prog_bar = ProgressBar(task_num, bar_width, file=file)
    results = []
    for task in tasks:
        results.append(func(task, **kwargs))
        prog_bar.update()
    prog_bar.file.write('\n')
    return results