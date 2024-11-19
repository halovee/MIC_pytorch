import os

def setup_dist(backend='nccl',rank = 0,world_size = 1,addr='localhost',port='29500'):
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)    # 从0开始
    os.environ['MASTER_ADDR'] = addr  # 在单机上是 'localhost'
    os.environ['MASTER_PORT'] = port  # 选择一个未被占用的端口

    from torch import distributed as dist
    dist.init_process_group(
        backend=backend,                          # or 'gloo' if NCCL is not available
        init_method=f'tcp://localhost:{port}',
        world_size=world_size,
        rank=rank)



