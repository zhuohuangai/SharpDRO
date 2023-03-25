import torch
from torchvision import datasets
import numpy as np 
from PIL import Image
import os


def get_cifar10_dataset(args, data_dir, indexs=None, is_train=False, transform=None):

    CIFAR = CIFAR10SSL
    cifar10_datasets = CIFAR(data_dir, indexs=indexs, train=is_train, transform=transform, download=True, dataset_name='cifar10')

    return cifar10_datasets

    
def get_cifar100_dataset(args, data_dir, indexs=None, is_train=False, transform=None):

    CIFAR = CIFAR100SSL
    cifar100_datasets = CIFAR(data_dir, indexs=indexs, train=is_train, transform=transform, download=True, dataset_name='cifar100')

    return cifar100_datasets


def get_cifar10_c_dataset(args, data_dir, indexs=None, is_train=False, transform=None, severity='1'):
    '''
    Returns:
        test_c_loader: corrupted testing set loader (original cifar10-C)
    CIFAR10-C has 50,000 test images. 
    The first 10,000 images in each .npy are of level 1 severity, and the last 10,000 are of level 5 severity.
    '''

    # # download:
    # url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
    # root_dir = data_dir
    # tgz_md5 = '56bf5dcef84df0e2308c6dcbcbbd8499'
    # if not os.path.exists(os.path.join(root_dir, 'CIFAR-10-C.tar')):
    #     download_and_extract_archive(url, root_dir, extract_root=root_dir, md5=tgz_md5)
    # elif not os.path.exists(os.path.join(root_dir, 'CIFAR-10-C')):
    #     extract_archive(os.path.join(root_dir, 'CIFAR-10-C.tar'), to_path=root_dir)

    CIFAR10_C = CIFAR10_C_SSL

    cifar10_c_datasets = CIFAR10_C(data_dir, indexs=indexs, train=is_train,
                 transform=transform, dataset_name='cifar10_c',
                 corruption=args.corruption, severity=severity)
    print('cifar10_c dataset for %s ready' % args.corruption)

    return cifar10_c_datasets


def get_cifar100_c_dataset(args, data_dir, indexs=None, is_train=False, transform=None, severity='1'):

    CIFAR100_C = CIFAR100_C_SSL

    cifar100_c_datasets = CIFAR100_C(data_dir, indexs=indexs, train=is_train,
                 transform=transform, dataset_name='cifar100_c',
                 corruption=args.corruption, severity=severity)
    print('cifar100_c dataset for %s ready' % args.corruption)

    return cifar100_c_datasets


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=False, dataset_name=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = torch.LongTensor(np.array(self.targets)[indexs])
        self.return_idx = return_idx
        self.dataset_name=dataset_name
        self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.targets_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets

    def __getitem__(self, index):
        img, target = self.data_index[index], self.targets_index[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.data_index)


class CIFAR100SSL(CIFAR10SSL):

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=False, dataset_name=None):
        super().__init__(root, indexs, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download, return_idx=return_idx, dataset_name=dataset_name)


class CIFAR10_C_SSL(datasets.VisionDataset):

    base_folder = "CIFAR-10-C"

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False, return_idx=False, dataset_name=None,
                 corruption = 'gaussian_noise', severity = '1'):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.corruption = corruption
        self.severity = severity

        if self.train:
            file_path = os.path.join(self.root, self.base_folder, 'train', self.severity)
        else:
            file_path = os.path.join(self.root, self.base_folder, 'test', self.severity)

        self.data = np.array(np.load(os.path.join(file_path, '%s.npy' % corruption)))
        self.targets = torch.LongTensor(np.load(os.path.join(file_path, 'labels.npy')))
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = self.targets[indexs]
        self.return_idx = return_idx
        self.dataset_name = dataset_name
        self.set_index()

    def set_index(self, indexes=None):
        if indexes is not None:
            self.data_index = self.data[indexes]
            self.targets_index = self.targets[indexes]
        else:
            self.data_index = self.data
            self.targets_index = self.targets

    def init_index(self):
        self.data_index = self.data
        self.targets_index = self.targets

    def __getitem__(self, index):
        img, target = self.data_index[index], self.targets_index[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.return_idx:
            return img, target
        else:
            return img, target, index

    def __len__(self):
        return len(self.data_index)


class CIFAR100_C_SSL(CIFAR10_C_SSL):

    base_folder = "CIFAR-100-C"

    def __init__(self, root, indexs=None, train=True, 
                transform=None, target_transform=None, 
                download=False, return_idx=False, dataset_name=None, 
                corruption='gaussian_noise', severity='1'):
        super().__init__(root, indexs, train, transform, target_transform, download, return_idx, dataset_name, corruption, severity)