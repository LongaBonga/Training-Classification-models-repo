from torchvision import datasets, transforms
from help_functions.distributed import num_distrib
import torch
import os
import numpy as np

class ToNumpy(transforms.ToTensor):
        """Convert Image in sample to Numpy."""
        def __call__(self, sample):
            return np.asarray(sample)[..., ::-1].transpose((2, 0, 1)) / 255
                    


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

    inference_transform = transforms.Compose([
    transforms.Resize(224),
    ])


    sampler_train = None
    sampler_val = None

    data_path_train = os.path.join(args.data_path, 'train')
    data_path_val = os.path.join(args.data_path, 'val')

    train_dataset = datasets.ImageFolder(data_path_train, transform= train_transform)
    val_dataset = datasets.ImageFolder(data_path_val,  transform= valid_transform)

    if num_distrib() > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
        sampler_val = torch.utils.data.distributed.DistributedSampler(val_dataset) 

    train_data = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            sampler=sampler_train)


    
    valid_data = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            sampler=sampler_val)
    return train_data, valid_data



def inference_loader(args):

    inference_transform = transforms.Compose([
    transforms.Resize(224),
    ToNumpy()
    ])

    data_path = os.path.join(args.data_path, 'val')

    inference_dataset = datasets.ImageFolder(data_path,  transform= inference_transform)

    inference_data = torch.utils.data.DataLoader(inference_dataset,
                                            batch_size=1,
                                            shuffle=True,
                                            num_workers=4)
    return inference_data


    # class ToTensor(object):
    # """Convert ndarrays in sample to Tensors."""

    # def __call__(self, sample):
    #     image, landmarks = sample['image'], sample['landmarks']

    #     # swap color axis because
    #     # numpy image: H x W x C
    #     # torch image: C x H x W
    #     image = image.transpose((2, 0, 1))
    #     return {'image': torch.from_numpy(image),
    #             'landmarks': torch.from_numpy(landmarks)}

    
