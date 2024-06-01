# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmcv/blob/7540cf73ac7e5d1e14d0ffbd9b6759e83929ecfc/mmcv/runner/dist_utils.py

import os
import subprocess

import torch
import torch.multiprocessing as mp
from torch import distributed as dist


def init_dist(launcher, backend='nccl', **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend, **kwargs):
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=None):
    """
    初始化slurm分布式训练环境。
    如果参数' ' port ' '未指定，则主端口将是system
    环境变量“MASTER_PORT”。如果' MASTER_PORT '不在系统中
    环境变量，那么将使用默认端口' ' 29500 ' '。
    参数:
        backend (str):火炬的后端。
        port (int，可选):主端口。默认为None。
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f'scontrol show hostname {node_list} | head -n1')
    # 指定主端口
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # 使用环境变量中的MASTER_PORT
    else:
        # 29500 是 torch.distributed 默认port
        os.environ['MASTER_PORT'] = '29500'
    # 如果环境变量MASTER_ADDR已经存在，则使用它
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def setup_for_distributed(is_master):
    """
        此功能在非主进程时禁用打印
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
