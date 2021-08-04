import argparse
import sys
import os.path as osp
import time

from train import train_func, val_func
from builders.optim_builder import build_optimizer
from builders.model_builder import build_model
from help_functions.distributed import print_at_master, to_ddp, reduce_tensor, num_distrib, setup_distrib, init_writer
from data_loading.data_loader import data_loader, inference_loader
from help_functions.flops_counter import get_model_complexity_info
from builders.model_builder import load_pretrained_weights

from OpenVino_inference.OpenVino_inference import eval_inference, conversion 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from collections import OrderedDict


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
    parser.add_argument("--model_path", type=str, default= None)
    parser.add_argument("--scheduler", action='store_true')
    parser.add_argument("--scheduler_coef", type=float, default = 2)
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--l1_coef", type=float, default=0.0000001)
    parser.add_argument("--complexity", action='store_true')
    parser.add_argument("--conversion", action='store_true')
    parser.add_argument("--eval_infr_path", type=str, default= None)



    args = parser.parse_args()
    setup_distrib(args)

    net = build_model(args.model).to(args.device)
    net = to_ddp(net, args)

    optimizer = build_optimizer(args, net, args.optimizer)
    criterion = nn.CrossEntropyLoss()

    train_data, valid_data = data_loader(args)
    
    if args.complexity:
        macs, params = get_model_complexity_info(net, (3, 224, 224),
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             )
        print_at_master('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print_at_master('{:<30}  {:<8}'.format('Number of parameters: ', params))

    start_time = time.time()

    if args.mode == "train":
        
        if args.model_path != None:
            if num_distrib() > 1:
                net.load_state_dict(torch.load(args.model_path))
            
            else:
                load_pretrained_weights(net, file_path = args.model_path)

                

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
        if num_distrib() > 1:

            net.load_state_dict(torch.load(args.model_path))

        else:
            load_pretrained_weights(net, file_path = args.model_path)

        acr = val_func(args, net, 
                criterion, 
                optimizer, 
                valid_data, 
                args.device, 
                None)

        print_at_master(f'Val acc: {acr * 100}')


    

    if args.conversion:
        conversion(args, net, args.model_path, (224, 224), save_path = args.output_dir)

    if args.eval_infr_path != None:

        inference_data = inference_loader(args)
        acr = eval_inference(args.eval_infr_path, inference_data)
        print_at_master(f'Inference acc: {acr * 100}')

    finish_time = time.time()
    print_at_master(f'Program finished! Total Time: {finish_time - start_time}')

if __name__ == "__main__":
    main()
