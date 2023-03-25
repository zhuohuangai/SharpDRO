import os
import argparse
import torch
from models import model_factory
from utils import Logger, CSVBatchLogger, log_args, log_data
from dataset.prepare_dataset import get_cifar_prepared_dataset, get_imagenet30_prepared_dataset, get_dataloader
from train import Trainer


def get_args():
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('-d', '--dataset', choices=['cifar10', 'cifar100', 'imagenet30'], default='cifar10')
    parser.add_argument('--desc', type=str, default='none')
    
    # Resume?
    parser.add_argument('--resume', default=False, action='store_true')
    
    # Data
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--augment_data', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('-s', '--severity', type=str, default='1')
    parser.add_argument('-c', '--corruption', choices=[
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'], default='gaussian_noise')
    parser.add_argument('--shift', default=True, action='store_false')
    parser.add_argument('-n', '--num_per_class', type=int, default=100)
    parser.add_argument('--num_val', type=int, default=2000)
    parser.add_argument('--num_test', type=int, default=2000)
    parser.add_argument('-p', '--proportion', type=float, default=0.2)
    parser.add_argument('--num_severity', type=int, default=5)

    # Objective
    parser.add_argument('--is_worst', default=True, action='store_false')

    # Model
    parser.add_argument('-m', '--model', choices=[
        'resnet18', 'resnet50', 'wideresnet'], default='wideresnet')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--is_pretrain', default=False, action='store_true')

    # Optimization
    parser.add_argument('--total_epoch', type=int, default=1000)
    parser.add_argument('--warmup_epoch', type=int, default=400)
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', default=True, action='store_false')
    parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--is_adaptive", default=False, action='store_true')
    parser.add_argument('--step_size', default=0.01, type=float)
    parser.add_argument('--distribution_agnostic', default=False, action='store_true')
    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_train', default=100, type=int)
    parser.add_argument('--log_eval', default=50, type=int)
    parser.add_argument('--save_step', default=50, type=int)
    parser.add_argument('--save_last', default=False, action='store_true')
    parser.add_argument('--save_best', default=False, action='store_true')

    return parser.parse_args()

    
def main():
    args = get_args()

    if args.dataset == 'cifar10':
        args.num_per_class = 4000
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_per_class = 400
        args.num_classes = 100
    elif args.dataset == 'imagenet30':
        args.num_per_class = 1000
        args.num_classes = 30
    args.desc = args.desc + '_dset' + args.dataset + '_sever' + args.severity + '_corr' + args.corruption

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # intialize device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger(os.path.join(args.log_dir, args.desc + '_log.txt'), mode='w')
    # Record args
    log_args(args, logger)
    # Set experiment seed
    # set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    logger.flush()

    # load dataset
    if args.dataset == 'imagenet30':
        train_dataset, val_dataset, test_dataset = get_imagenet30_prepared_dataset(args)
    else:
        train_dataset, val_dataset, test_dataset = get_cifar_prepared_dataset(args)
    train_loader, val_loader_tuple, test_loader_tuple = get_dataloader(args, train_dataset, val_dataset, test_dataset)

    # log dataset
    logger.write('Training Labeled Data...\n')
    log_data(train_dataset, logger)
    for i in range(args.num_severity+1):
        logger.write(f'Validation Data {i:d}...\n')
        log_data(val_dataset[i], logger)
        logger.write(f'Test Data {i:d}...\n')
        log_data(test_dataset[i], logger)


    train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, args.desc + '_train.csv'), train_dataset.num_distribution, mode='w')
    val_clean_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, args.desc + '_val_clean.csv'), val_dataset[0].num_distribution, mode='w')
    val_logger_tuple = [val_clean_csv_logger]
    for i in range(1, args.num_severity+1):
        val_logger_tuple.append(CSVBatchLogger(os.path.join(args.log_dir, args.desc + f'_val_robust_{i:d}.csv'), val_dataset[i].num_distribution, mode='w'))
    val_logger_tuple = tuple(val_logger_tuple)
    test_clean_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, args.desc + '_test_clean.csv'), test_dataset[0].num_distribution, mode='w')
    test_logger_tuple = [test_clean_csv_logger]
    for i in range(1, args.num_severity+1):
        test_logger_tuple.append(CSVBatchLogger(os.path.join(args.log_dir, args.desc + f'_test_robust_{i:d}.csv'), test_dataset[i].num_distribution, mode='w'))
    test_logger_tuple = tuple(test_logger_tuple)

    model = model_factory.get_network(args.model)(classes=args.num_classes)

    trainer = Trainer(device, model, logger, train_csv_logger, val_logger_tuple, test_logger_tuple, train_dataset.num_distribution)
    trainer.train(args, train_loader, val_loader_tuple, test_loader_tuple)

    train_csv_logger.close()
    for i in range(args.num_severity+1):
        val_logger_tuple[i].close()
    for i in range(args.num_severity+1):
        test_logger_tuple[i].close()

if __name__=='__main__':
    main()