import argparse
import sys
import os.path as osp
import time

from train import train_func
from builders.optim_builder import build_optimizer
from builders.model_builder import build_model
from torch.utils.tensorboard import SummaryWriter
from help_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models



def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='image classification params')
    parser.add_argument('--output_dir', type=str, default='', help='directory to store training artifacts')
    parser.add_argument('--model', type=str, default='mobilenet_v2', help='name of model')
    parser.add_argument('--optimazer', type=str, default='sgd', help='name of optimazer')
    parser.add_argument('--batch_size', type=int, default='64', help='batch size')
    parser.add_argument('--mode', type=str, default='train', help='mode of model train/evaluation')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda','cpu'],
                        help='choose device to train on')
    parser.add_argument("--local_rank", default=0, type=int)


    args = parser.parse_args()

    net = build_model(args.model).to(args.device) # make choice of models
    net = to_ddp(net, args)

    optimizer = build_optimizer(net, args.optimazer)
    criterion = nn.CrossEntropyLoss()


    # data train_data, val_data processing going there
    train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    ])

    valid_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    ])


    cifar100_train = datasets.CIFAR100('path/to/cifar100_root/', train = True, download=True, transform= train_transform)
    train_data = torch.utils.data.DataLoader(cifar100_train,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=4)


    cifar100_valid = datasets.CIFAR100('path/to/cifar100_root/', train = False, download=True, transform= valid_transform)
    valid_data = torch.utils.data.DataLoader(cifar100_valid,
                                              batch_size=32,
                                              shuffle=True,
                                              num_workers=4)

    # if cfg.regime.type == "evaluation":
    #     # function for eval
    # else:
    writer = SummaryWriter(args.output_dir, comment = args.model)

    if args.mode == "train":
        train_func(args, net,
                criterion,
                optimizer,
                train_data,
                valid_data,
                args.output_dir,
                args.model,
                args.device,
                writer=writer)


if __name__ == "__main__":
    main()
