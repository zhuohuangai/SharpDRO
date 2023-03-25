'''
Tiny-ImageNet:
Download by wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
Run python create_tin_val_folder.py to construct the validation set. 

Tiny-ImageNet-C:
Download by wget https://zenodo.org/record/2469796/files/TinyImageNet-C.tar?download=1
Run python dataloaders/fix_tin_c.py to remove the redundant images in TIN-C.

Tiny-ImageNet-V2:
Download ImageNet-V2 from http://imagenetv2public.s3-website-us-west-2.amazonaws.com/
Run python dataloaders/construct_tin_v2.py to select 200-classes from the full ImageNet-V2 dataset.

https://github.com/snu-mllab/PuzzleMix/blob/master/load_data.py
'''

import torch.utils.data as data
from torchvision import datasets
from PIL import Image
import os
import os.path
import numpy as np


def get_imagenet_30_datasets(args, image_list, is_train=False, transform=None):

    if is_train:
        path_prefix = args.data_dir + '/ImageNet-30-64/train'
    else:
        path_prefix = args.data_dir + '/ImageNet-30-64/test'

    # in30_write_path = ImageFolder(root="./data/ImageNet-30/test", transform=transform, fname='ImageNet-30-test-path.txt')

    # input("path written")

    in30_datasets = MyImageFolder(image_list, path_prefix, transform=transform, dataset_name='imagenet30')

    return in30_datasets


def get_imagenet_30_c_datasets(args, image_list, is_train=False, transform=None, severity='1'):

    if is_train:
        path_prefix = args.data_dir + '/ImageNet-30-64-C/train/' + args.corruption + '/' + severity
    else:
        path_prefix = args.data_dir + '/ImageNet-30-64-C/test/' + args.corruption + '/' + severity

    in30_c_datasets = MyImageFolder(image_list, path_prefix, transform=transform, dataset_name='imagenet30_c')

    return in30_c_datasets

def default_loader(path):
    return Image.open(path).convert('RGB')


def make_dataset_nolist(image_list, path_prefix):
    with open(image_list) as f:
        image_index = [os.path.join(path_prefix, x.split(' ')[0]) for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


class MyImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, path_prefix, transform=None, target_transform=None, return_paths=False, dataset_name=None,
                 loader=default_loader,train=False, return_id=False):
        imgs, labels = make_dataset_nolist(image_list, path_prefix)
        self.imgs = imgs
        self.labels= labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.return_paths = return_paths
        self.return_id = return_id
        self.train = train
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.imgs[index]
        target = self.labels[index]
        img = self.loader(path)
        img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_paths:
            return img, target, path
        elif self.return_id:
            return img, target ,index
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)

class ImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, return_idx=False, dataset_name=None, fname=None):
        super().__init__(root, transform)
        self.return_idx = return_idx
        self.dataset_name = dataset_name
        self.return_idx = return_idx

        self.fname = fname
        # For writing path.txt purposes.
        if self.fname is not None:
            for i in range(len(self.samples)):
                with open(self.fname, 'a') as f:
                    f.writelines(self.samples[i][0] + ' ' + str(self.samples[i][1]) + '\n')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_idx:
            return sample, target
        else:
            return sample, target, index


            