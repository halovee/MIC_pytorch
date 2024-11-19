import argparse
import os
import time
import torch
import configs.gtaHR2csHR_mic_hrda_s2_8e10a as gtaHR2csHR_mic_hrda_s2_8e10a
from configs import get_cfg_data, get_cfg_model, get_cfg_run
from data.builder_dataloader import build_dataloader
from data.datasets.cityscapes import CityscapesDataset
from data.datasets.gta import GTADataset
from data.datasets.uda_dataset import UDADataset
from model.runner.hooks import EvalHook, CheckpointHook, PolyLrUpdater, DistEvalHook
from model.runner.hooks.log_buffer import LogBuffer
from model.runner.hooks.text_logger import TextLoggerHook
from model.runner.optimizer import build_optimizer
from model.runner.schedules import set_random_seed
from model.segmentors.dacs import DACS
from utils import env_info, get_time_str, get_dist_info, mkdir_or_exist
from utils.dist import MMDataParallel
from utils.logger import get_root_logger

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
        A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Segformer Network")
    parser.add_argument("--gpus", type=int, default=1, help="choose number of gpu devices to use")
    parser.add_argument("-c", "--config", type=str, default='./configs/configUDA_gta.json', help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None, help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default="GTA2City", help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None, help='Include to save images (default: None)')
    return parser.parse_args()

def main():
    # 环境信息及配置
    cfg_model = get_cfg_model('model')
    cfg_data = get_cfg_data('data')
    cfg_run = get_cfg_run('cfg_run')

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())

    torch.cuda.empty_cache()

    seed = cfg_run['seed']
    set_random_seed(seed)
    cfg = gtaHR2csHR_mic_hrda_s2_8e10a.cfg
    time_stamp = get_time_str()
    print(f'当前时间：{time_stamp}\n环境信息：{env_info()}\n随机种子：{seed}\n')

    # log文件
    if resume:
        work_dir = os.path.join(*resume.split('/')[:-1] + '_resume-' + time_stamp)
    else:
        work_dir = os.path.join('./saved', timestamp + '-' + name)
    if not os.path.isabs(work_dir):                # 判断路径是否为绝对路径
        work_dir = os.path.abspath(str(work_dir))  # 转换为绝对路径
    work_dir = os.path.abspath(str(work_dir))
    if not os.path.exists(work_dir):
        mkdir_or_exist(work_dir)

# 1、构建模型
    cfg_model['train_cfg'].update({'work_dir': work_dir})
    cfg_model['train_cfg'].update({'log_config': cfg_run['log_config']})
    model = DACS(**cfg['model'])
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of_{model.__class__.__name__}_ trainable parameters: {num_parameters}")
    model.init_weights()
    log_file = os.path.join(work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')
    logger.info(model)
    # load_checkpoint(model, 'pretrained/latest.pth', map_location='cpu', strict=False)

# 2、构建训练数据集
    cfg_data_train = cfg_data['train']
    cfg_gta = cfg_data_train['source_gta']
    gta_dataset = GTADataset(**cfg_gta)
    cfg_city = cfg_data_train['target']
    city_dataset = CityscapesDataset(**cfg_city)
    uda_dataset = UDADataset(gta_dataset, city_dataset, cfg_data_train)

# 3、构建训练数据集加载器
    batch_size = cfg_data['samples_per_gpu']
    num_workers = cfg_data['workers_per_gpu']
    uda_data_loader = build_dataloader(
        dataset=uda_dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=num_workers,
        # cfg.gpus will be ignored if distributed
        num_gpus=len(gpu_ids),
        dist=len(gpu_ids) > 1,
        seed=seed,
        drop_last=True)

# 4、放置模型到GPU上，模型要被包装在MMDataParallel中，以解包数据集DataContainers类
    model = MMDataParallel(model.cuda(gpu_ids[0]), device_ids=gpu_ids)

# 5、构建优化器
    cfg_optimizer = cfg_run['optimizer']
    optimizer = build_optimizer(model, cfg_optimizer)

# 6、构建运行时参数
    _rank, _world_size = get_dist_info()
    mode = 'train'
    _hooks = []
    _epoch = 0
    _iter = 0
    _inner_iter = 0
    _max_epochs = None
    _max_iters = cfg_run['runner']['max_iters']
    log_buffer = LogBuffer()

# 7、构建验证集、验证数据加载器
    cfg_data_val = cfg_data['val']
    cfg_data_val.update({'test_mode': True})
    city_val_dataset = CityscapesDataset(**cfg_data_val)
    val_data_loaders = build_dataloader(
        dataset=city_val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=num_workers,
        # cfg.gpus will be ignored if distributed
        dist=len(gpu_ids) > 1,
        shuffle=False)

    # 构建PolyLrUpdaterHook （VERY_HIGH）
    cfg_lr = cfg_run['lr_config']
    cfg_lr.pop('policy')
    lr_scheduler = PolyLrUpdater(**cfg_lr)

    # 构建CheckpointHook  （NORMAL）
    meta = dict(CLASSES=uda_dataset.CLASSES, PALETTE=uda_dataset.PALETTE)
    cfg_checkpoint = cfg_run['checkpoint_config']
    cfg_checkpoint.update({'meta': meta})
    checkpoint = CheckpointHook(**cfg_checkpoint)

    # 构建TextLoggerHook （VERY_LOW）
    cfg_textlogger = cfg_run['log_config']
    text_logger = TextLoggerHook(**cfg_textlogger)

    # 构建EvalHook （NORMAL）
    cfg_eval = cfg_run['evaluation']
    cfg_eval.update({'by_epoch': cfg_run['runner']['type'] != 'IterBasedRunner'})
    eval_hook = DistEvalHook(val_data_loaders, **cfg_eval) if len(gpu_ids)>1 else EvalHook(val_data_loaders, **cfg_eval)

    # 注册训练钩子
    _hooks = [lr_scheduler, checkpoint, eval_hook, text_logger]

# 8、训练迭代过程

    # before_run
    lr_scheduler.before_run(optimizer)
    checkpoint.before_run(work_dir)
    eval_hook.before_run(meta)
    text_logger.before_run(meta, work_dir, timestamp, _iter)
    # before_run

    # 构建数据集迭代器
    uda_loader_iter = iter(uda_data_loader)

    # before_epoch
    t = time.time()
    text_logger.before_epoch(log_buffer)
    # before_epoch

    while _iter < _max_iters:
        # 设置模型模式为训练
        model.train()
        mode = 'train'
        # 迭代数据batch
        try:
            uda_data_batch= next(uda_loader_iter)
        except StopIteration:
            _epoch += 1
            print('Epochs since start: ',_epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            uda_loader_iter = iter(uda_data_loader)
            uda_data_batch= next(uda_loader_iter)

        # before_train_iter
        lr_scheduler.before_train_iter(_epoch, _max_epochs, _iter, _max_iters, optimizer)
        eval_hook.before_train_iter(model, log_buffer, _epoch, _iter, logger, work_dir, meta)
        log_buffer.update({'data_time': time.time() - t})
        # before_train_iter

        # 模型训练
        outputs = model.train_step(uda_data_batch, optimizer)
        if 'log_vars' in outputs:
            log_buffer.update(outputs['log_vars'], outputs['num_samples'])

        # after_train_iter
        checkpoint.after_train_iter(meta, _epoch, _iter, _max_iters, model)
        eval_hook.after_train_iter(model, log_buffer, _epoch, _iter, logger, work_dir, meta)
        log_buffer.update({'time': time.time() - t})
        t = time.time()
        text_logger.after_train_iter(log_buffer, optimizer, meta, mode, _iter, _epoch, _inner_iter, logger, _max_iters, uda_data_loader, model)

        _inner_iter += 1
        _iter += 1
        # after_train_iter
    time.sleep(1)
    # after_train_epoch
    checkpoint.after_train_epoch(meta, _iter, _epoch, _max_epochs, model)
    eval_hook.after_train_epoch(model, log_buffer, _epoch, _iter, logger, work_dir, meta)
    text_logger.after_train_epoch(log_buffer, optimizer, meta, mode, _iter, _epoch, _inner_iter, logger, _max_iters, uda_data_loader, model)
    # after_train_epoch

    # after_run
    # after_run



if __name__ == '__main__':

    print('\n---------------------------------Starting---------------------------------\n')
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args = get_arguments()
    resume = args.resume
    name = args.name

    gpu_ids = range(0, 1)     #　选择使用的GPU数量

    main()

