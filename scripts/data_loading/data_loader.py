from torchvision import datasets, transforms
from help_functions.distributed import num_distrib
import torch
import os

def data_loader(args):
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


    sampler_train = None
    sampler_val = None

    data_path_train = os.path.join(args.data_path, 'train')
    data_path_val = os.path.join(args.data_path, 'val')

    train_dataset = datasets.CIFAR100(data_path_train, transform= train_transform)
    val_dataset = datasets.CIFAR100(data_path_val,  transform= valid_transform)

    if num_distrib() > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
        sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset) 

    train_data = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            sampler=sampler_train)


    
    valid_data = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=32,
                                            shuffle=False,
                                            num_workers=4,
                                            sampler=sampler_val)
    return train_data, valid_data