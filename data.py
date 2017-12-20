import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms

_DATASETS_MAIN_PATH = '/home/Datasets'
_dataset_path = {
    'cifar10': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR10'),
    'cifar100': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR100'),
    'stl10': os.path.join(_DATASETS_MAIN_PATH, 'STL10'),
    'mnist': os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
    'imagenet': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/train'),
        'val': os.path.join(_DATASETS_MAIN_PATH, 'ImageNet/val')
    }
}


def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True):
    train = (split == 'train')
    if name == 'cifar10':
        return datasets.CIFAR10(root=_dataset_path['cifar10'],
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=_dataset_path['cifar100'],
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'imagenet':
        path = _dataset_path[name][split]
        return datasets.ImageFolder(root=path,
                                    transform=transform,
                                    target_transform=target_transform)
