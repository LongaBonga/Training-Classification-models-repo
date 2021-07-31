import os
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

def get_dist_info():
    initialized = dist.is_available() and dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def is_master():
    rank, _ = get_dist_info()
    return rank == 0


def print_at_master(str):
    if is_master():
        print(str)

def add_to_writer(writer, loss, pred, size, cnt, mode):
    if is_master():
        if mode == 'Train':
            writer.add_scalar(mode +' loss:', (loss / size), cnt)
            writer.add_scalar(mode +' acc:', (pred / size) * 100, cnt)
        else:
            writer.add_scalar(mode +' acc:', (pred / size) * 100, cnt)


def setup_distrib(args):
    if num_distrib() > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')


def to_ddp(model, args):
    if num_distrib() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    return model


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= n
    return rt


def num_distrib():
    return int(os.environ.get('WORLD_SIZE', 0))

def init_writer(args):
    if is_master():
        return SummaryWriter(args.output_dir, comment = args.model)