import os
import json
import torch
import argparse
import numpy as np
from utils import *
from losses import *
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from evaluate import pgd_evaluate


parser = argparse.ArgumentParser()
parser.add_argument("--mnist", action="store_true", default=False)
parser.add_argument("--cifar", action="store_true", default=False)
parser.add_argument("--loss", type=str, choices=["r", "as", "mt"], default="r")
parser.add_argument('--batch-size', type=int, default=100, metavar='N')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--beta', default=1, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--period", type=int, default=10)
parser.add_argument("--max", type=float, default=6)
parser.add_argument('--weight-decay', '--wd',
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument("--r1", type=int, default=60)
parser.add_argument("--r2", type=int, default=75)
parser.add_argument("--r3", type=int, default=90)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--train-epsilon', type=float,
                    help='perturbation')
parser.add_argument('--train-num-steps', type=int,
                    help='perturb number of steps')
parser.add_argument('--train-step-size', type=float,
                    help='perturb step size')
parser.add_argument('--test-epsilon', type=float,
                    help='perturbation')
parser.add_argument('--test-num-steps', type=int,
                    help='perturb number of steps')
parser.add_argument('--test-step-size', type=float,
                    help='perturb step size')
parser.add_argument('--random', default=True,
                    help='random initialization for PGD')
parser.add_argument('--log-interval', type=int, default=100, metavar='N')
parser.add_argument('--model-folder', default="checkpoints",
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument("--log-folder", type=str, default="log")
parser.add_argument("--result-folder", type=str, default="results")
parser.add_argument("--data-root", default="~/data", type=str)
parser.add_argument("--cuda", type=int, default=None)
parser.add_argument("--seed", type=int, default=0,
                    help="set to -1 to use random without a seed")
parser.add_argument("--net", type=str,
                    choices=["wideres", "resnet", "nn", "resnist"])
parser.add_argument("--save-name", type=str, required=True)
args = parser.parse_args()

check_args(args)

device, kwargs = set_device(args)
set_hyper(args)
train_loader, test_loader = load_dataset(args)


def main(args):
    # init model, ResNet18() can be also used here for training
    model = set_network(args, device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    model_folder = os.path.join(args.model_folder, args.save_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    if not os.path.exists(args.result_folder):
        os.mkdir(args.result_folder)
    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)

    if args.loss == "r":
        loss_func = RobustLoss(args)
    elif args.loss == "as":
        loss_func = AccurateStableLoss(args)
    elif args.loss == "mt":
        loss_func = MultiTaskLoss(args)
    else:
        raise ValueError("How come you reach here?")

    logger = create_logger(args.log_folder, args.save_name, "info")
    result_name = save_args(args, logger)

    max_rob = 0.0

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        # adjust_reg_param(args, epoch)

        # logger.info("Epoch {}\tbeta {:.2f}".format(epoch, args.beta))
        logger.info("Epoch {}".format(epoch))

        # adversarial training per epoch
        losses = []
        accurate_losses = []
        stable_losses = []
        total = 0
        tr_acc = 0
        tr_sta = 0  # training stable
        tr_rob = 0  # training robust

        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            loss, ret = loss_func(model, x, y)

            accurate_losses.append(ret["l_a"].item())
            stable_losses.append(ret["l_s"].item())
            losses.append(loss.item())

            total += len(y)
            tr_acc += ret["accurate"]
            tr_sta += ret["stable"]
            tr_rob += ret["robust"]

            loss.backward()
            optimizer.step()

            # print progress
            if (batch_idx + 1) % args.log_interval == 0:
                logger.info(
                    '[{:.0f}%]\tL: {:.4f}  A: {:.4f}  S: {:.4f}  '.format(
                        (batch_idx * args.batch_size + len(x)) /
                        len(train_loader.dataset) * 100,
                        np.mean(losses), np.mean(accurate_losses),
                        np.mean(stable_losses),
                        )
                    )
                accurate_losses = []
                stable_losses = []
                losses = []
        # end of adversarial training per epoch

        # evaluation on natural examples
        logger.info('=' * 80)
        rate_tr_acc = tr_acc / total
        rate_tr_sta = tr_sta / total
        rate_tr_rob = tr_rob / total
        rate_te_acc, rate_te_sta, rate_te_rob = pgd_evaluate(
            model, test_loader, args.test_epsilon,
            args.test_num_steps, args.test_step_size, device
            )
        logger.info(
            "Train A: {:.2f}%, S: {:.2f}%, R: {:.2f}%\t"
            "Test A: {:.2f}%, S: {:.2f}%, R: {:.2f}%".format(
                100 * rate_tr_acc, 100 * rate_tr_sta, 100 * rate_tr_rob,
                100 * rate_te_acc, 100 * rate_te_sta, 100 * rate_te_rob
                ))
        logger.info('=' * 80)

        if rate_te_rob > max_rob:
            max_rob = rate_te_rob

        with open(result_name, "r") as f:
            dic = json.load(f)
        dic["accuracy"][epoch] = rate_te_acc
        dic["robustness"][epoch] = rate_te_rob

        with open(result_name, "w") as f:
            json.dump(dic, f, indent=4)
        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(
                model.state_dict(),
                os.path.join(model_folder, 'model-{}-epoch{}.pt'.format(
                    args.net, epoch)))
            torch.save(
                optimizer.state_dict(),
                os.path.join(model_folder, 'opt-{}-epoch{}.tar'.format(
                    args.net, epoch)))

    best = "_{}".format(int(max_rob * 10000))
    os.rename(result_name, result_name[:-5] + best + result_name[-5:])


if __name__ == "__main__":
    main(args)
