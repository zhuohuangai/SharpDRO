import sys
import os
from cv2 import resize
import torch
import numpy as np
import csv
import cv2
from PIL import Image

def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x.detach()
    

class Logger(object):
    def __init__(self, fpath=None, mode='w'):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, mode)

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

class CSVBatchLogger:
    def __init__(self, csv_path, num_distribution, mode='w'):
        self.columns = ['epoch', 'batch']
        self.columns.append('avg_acc')
        self.columns.append('avg_loss')
        for idx in range(num_distribution):
            self.columns.append(f'processed_distribution_counts_{idx}')
            self.columns.append(f'distribution_loss_{idx}')
            self.columns.append(f'adv_prob_{idx}')
            self.columns.append(f'distribution_acc_{idx}')
            self.columns.append(f'gradnorm_{idx}')
        self.columns.append('model_norm_sq')
        self.columns.append('reg_loss')

        self.path = csv_path
        self.file = open(csv_path, mode)
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns)
        if mode=='w':
            self.writer.writeheader()

    def log(self, epoch, batch, stats_dict):
        stats_dict['epoch'] = epoch
        stats_dict['batch'] = batch
        self.writer.writerow(stats_dict)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def log_args(args, logger):
    for argname, argval in vars(args).items():
        logger.write(f'{argname.replace("_"," ").capitalize()}: {argval}\n')
    logger.write('\n')


def log_data(data, logger):
    for distribution_idx in range(data.num_distribution):
        logger.write(f'    {data.distribution_str[distribution_idx]}: n = {data.distribution_counts[distribution_idx]:.0f}\n')


def tensor2im(input_image, mean, std, imtype=np.uint8, color_map=False):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255  # grayscale to RGB
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:
        image_numpy = input_image
    if color_map:
        image_numpy = cv2.applyColorMap(image_numpy.astype(imtype), cv2.COLORMAP_JET)
    image_numpy = Image.fromarray(image_numpy.astype(imtype)).resize((128, 128), resample=Image.NEAREST).resize((256, 256))
    return image_numpy