import os
import torch
import argparse
import numpy as np
import torchvision
from utils import set_device
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


def load_dataset(args):
    if args.cifar:
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])
        trainset = torchvision.datasets.CIFAR10(
            root=args.data_root, train=True, download=True,
            transform=transform_train
            )
        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True
            )
        testset = torchvision.datasets.CIFAR10(
            root=args.data_root, train=False, download=True,
            transform=transform_test
            )
        test_loader = DataLoader(
            testset, batch_size=args.batch_size, shuffle=False
            )
    else:
        train_loader = DataLoader(
            datasets.MNIST(args.data_root, train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True)

        test_loader = DataLoader(
            datasets.MNIST(args.data_root, train=False,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader


def load_shuffle(X_path, Y_path):
    X = np.load(X_path)
    Y = np.load(Y_path)

    if args.cifar:
        n_train = 50000
    elif args.mnist:
        n_train = 60000
    else:
        n_train = None

    assert n_train + 10000 == X.shape[0]

    random_index = np.arange(X.shape[0])
    print("Shuffling...")
    np.random.shuffle(random_index)
    train_index = random_index[ : n_train]
    test_index = random_index[n_train : ]
    X_train = X[train_index]
    Y_train = Y[train_index]
    X_test = X[test_index]
    Y_test = Y[test_index]

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true")
    parser.add_argument("--cifar", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int)
    parser.add_argument("--data-root", default=".", type=str)
    args = parser.parse_args()

    save_folder = "shuffled"
    X_path = os.path.join(save_folder, "X.npy")
    Y_path = os.path.join(save_folder, "Y.npy")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if not os.path.exists(X_path):
        X = []
        Y = []
        train_loader, test_loader = load_dataset(args)

        for x, y in train_loader:
            X.append(x.numpy())
            Y.append(y.numpy())

        for x, y in test_loader:
            X.append(x.numpy())
            Y.append(y.numpy())

        print("Concatenating X...")
        X = np.concatenate(X, axis=0)
        print("Concatenating Y...")
        Y = np.concatenate(Y, axis=0)
        np.save(os.path.join(save_folder, "X"), X)
        np.save(os.path.join(save_folder, "Y"), Y)

    X_train, Y_train, X_test, Y_test = load_shuffle(X_path, Y_path)
    data = [X_train, Y_train, X_test, Y_test]
    names = ["X_train", "Y_train", "X_test", "Y_test"]
    for datum, name in zip(data, names):
        np.save(os.path.join(save_folder, name), datum)
