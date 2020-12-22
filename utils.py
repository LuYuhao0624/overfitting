import os
import json
import torch
import logging
import torchvision
import numpy as np
from models.resnet import *
from models.small_cnn import *
from models.net_mnist import *
from models.wideresnet import *
from torchvision import datasets, transforms


def create_logger(save_path='', name="", level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO
    else:
        raise ValueError("level argument only accepts debug/info")

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, name + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger


def adjust_reg_param(args, epoch):
    if args.period is None:
        return
    max_beta = None
    if args.max is not None:
        max_beta = args.max
    if (epoch - 1) % args.period == 0 and epoch != 1:
        beta = args.beta + args.alpha
        if max_beta is not None and beta > max_beta:
            setattr(args, "beta", max_beta)
        else:
            setattr(args, "beta", beta)


def adjust_learning_rate(args, optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= args.r1:
        lr = args.lr * 0.1
    elif epoch >= args.r2:
        lr = args.lr * 0.01
    elif epoch >= args.r3:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def select_gpu(cuda):
    if cuda is not None:
        torch.cuda.set_device(cuda)


def set_device(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    select_gpu(args.cuda)
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
    return device, kwargs


def set_hyper(args):
    attrs = [
        "lr", "net", "weight_decay",
        "train_epsilon", "train_num_steps", "train_step_size",
        "test_epsilon", "test_num_steps", "test_step_size"
        ]
    if args.cifar:
        default_values = [0.01, "resnet", 1e-4, 0.031, 10, 0.007, 0.031, 20,
                          0.003]
    elif args.mnist:
        default_values = [0.01, "nn", 0, 0.3, 40, 0.01, 0.3, 40, 0.01]
    else:
        raise ValueError("How come you reach here?")
    for i, attr in enumerate(attrs):
        if getattr(args, attr) is None:
            setattr(args, attr, default_values[i])


def save_args(args, logger):
    result_name = os.path.join(args.result_folder, args.save_name + ".json")

    important_args = [
        "mnist", "cifar", "net", "loss", "batch_size", "epochs", "beta",
        # "alpha", "period", "max",
        "weight_decay", "lr", "r1", "r2", "r3", "momentum",
        "train_epsilon", "train_num_steps", "train_step_size",
        "test_epsilon", "test_num_steps", "test_step_size",
        "model_folder", "log_folder", "result_folder",
        # "attack_folder",
        "seed", "save_name"
        # , "save_attack", "load_data"
        ]

    dic = {}

    for arg in important_args:
        value = getattr(args, arg, None)
        num = (23 - len(arg)) // 8
        logger.info("{}".format(arg) + "\t" * num + ": {}".format(value))
        dic[arg] = value

    dic["accuracy"] = {}
    dic["robustness"] = {}

    with open(result_name, "w") as f:
        json.dump(dic, f, indent=4)

    return result_name


def load_dataset(args):
    if args.cifar:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])
        trainset = torchvision.datasets.CIFAR10(
            root=args.data_root, train=True, download=True,
            transform=transform_train
            )
        testset = torchvision.datasets.CIFAR10(
            root=args.data_root, train=False, download=True,
            transform=transform_test
            )
    else:
        trainset = datasets.MNIST(args.data_root, train=True, download=True,
                                  transform=transforms.ToTensor())

        testset = datasets.MNIST(args.data_root, train=False,
                                 transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    return train_loader, test_loader


def check_args(args):
    if args.mnist & args.cifar is True:
        raise ValueError("Cannot use --mnist and --cifar at the same time.")

    if args.mnist | args.cifar is False:
        raise ValueError("Specify one of --mnist and --cifar.")


def set_network(args, device):
    if args.cifar:
        if args.net == "wideres":
            model = WideResNet().to(device)
        else:
            model = ResNet18().to(device)
    elif args.mnist:
        if args.net == "nn":
            model = SmallCNN().to(device)
        else:
            model = ResNet18MNIST().to(device)
    else:
        raise ValueError("How come you reach here?")
    return model
