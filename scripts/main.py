import argparse
import sys
import os.path as osp
import time

from train import train_func, val_func
from builders.optim_builder import build_optimizer
from builders.model_builder import build_model
from help_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib, init_writer
from data_loading.data_loader import data_loader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models




def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='image classification params')
    parser.add_argument('--output_dir', type=str, default='', help='directory to store training artifacts')
    parser.add_argument('--model', type=str, default='mobilenet_v2', help='name of model')
    parser.add_argument('--optimizer', type=str, default='sgd', help='name of optimazer')
    parser.add_argument('--batch_size', type=int, default='64', help='batch size')
    parser.add_argument('--mode', type=str, default='train', help='mode of model train/evaluation')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda','cpu'],
                        help='choose device to train on')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--fp16",  action='store_true')
    parser.add_argument("--data_path", type=str, default='path/to/cifar100_root/')
    parser.add_argument("--model_path", type=str, default='.')
    parser.add_argument("--scheduler_coef", type=float, default=0.97)
    parser.add_argument("--num_epoch", type=int, default=40)



    args = parser.parse_args()
    setup_distrib(args)

    net = build_model(args.model).to(args.device)
    net = to_ddp(net, args)

    optimizer = build_optimizer(net, args.optimizer)
    criterion = nn.CrossEntropyLoss()

    train_data, valid_data = data_loader(args)
    
    

    if args.mode == "train":

        writer = init_writer(args)
        train_func(args, net,
                criterion,
                optimizer,
                train_data,
                valid_data,
                args.output_dir,
                args.model,
                args.device,
                writer=writer)

    if args.mode == "val":
        net.load_state_dict(torch.load(args.model_path))
        loss, acr = val_func(args, net, 
                criterion, 
                optimizer, 
                valid_data, 
                args.device, 
                None, 
                epoch = 0)

        print_at_master(f'Val loss: {loss}')
        print_at_master(f'Val acc: {acr * 100}')

if __name__ == "__main__":
    main()
