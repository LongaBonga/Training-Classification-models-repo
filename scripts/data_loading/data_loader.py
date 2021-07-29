from torchvision import datasets, transforms
from help_functions.distributed import num_distrib
import torch

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

    if num_distrib() > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
        sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset) 

    cifar100_train = datasets.CIFAR100(args.data_path, train = True, download=True, transform= train_transform)
    train_data = torch.utils.data.DataLoader(cifar100_train,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            sampler=sampler_train)


    cifar100_valid = datasets.CIFAR100(args.data_path, train = False, download=True, transform= valid_transform) 
    valid_data = torch.utils.data.DataLoader(cifar100_valid,
                                            batch_size=32,
                                            shuffle=True,
                                            num_workers=4,
                                            sampler=sampler_val)
    return train_data, valid_data