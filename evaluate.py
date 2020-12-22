import torch
import argparse
import numpy as np
import torch.nn as nn
from attacks import *
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from utils import set_device, set_network, load_dataset


def pgd_evaluate(model, data_loader, epsilon, num_steps, step_size):
    model.eval()
    accurate = 0
    stable = 0
    robust = 0
    total = 0

    for i, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        x_adv = PGD()(x, model, epsilon, num_steps, step_size, y, True)
        x_logit = model(x)
        x_plabel = torch.max(x_logit, dim=1)[1]
        x_adv_logit = model(x_adv)
        x_adv_plabel = torch.max(x_adv_logit, dim=1)[1]
        accurate += (x_plabel == y).float().sum()
        stable += (x_adv_plabel == x_plabel).float().sum()
        robust += (x_adv_plabel == y).float().sum()
        total += len(y)

    return accurate / total, stable / total, robust / total



def _pgd(model, X, y, epsilon, num_steps, step_size, random, device):
    out = model(X)
    acc = (out.data.max(1)[1] == y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(
            -epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    rob = (model(X_pgd).data.max(1)[1] == y.data).float().sum()
    adv = X_pgd.cpu().detach().numpy()
    eta = (X_pgd - X).cpu().detach().numpy()
    return acc.item(), rob.item(), adv, eta, X_pgd


# def _pgd(model, X, y, epsilon, num_steps, step_size, args, device):
#     model.eval()
#     X_pgd = X + 0.001 * torch.randn(X.shape).cuda()
#     X_pgd = torch.clamp(X_pgd, 0, 1)
#     # generate adv
#     for _ in range(num_steps):
#         X_pgd.requires_grad_()
#         with torch.enable_grad():
#             loss_adv = F.cross_entropy(model(X_pgd), y)
#         grad = torch.autograd.grad(loss_adv, [X_pgd])[0]
#         X_pgd = X_pgd.detach() + step_size * torch.sign(grad.detach())
#         X_pgd = torch.min(torch.max(X_pgd, X - epsilon), X + epsilon)
#         X_pgd = torch.clamp(X_pgd, 0.0, 1.0)
#     # if args.train_mode_eval:
#     #     model.train()
#     model.train()
#     X_pgd = Variable(torch.clamp(X_pgd, 0.0, 1.0), requires_grad=False)
#     X_logit = model(X)
#     X_plabel = torch.max(X_logit, dim=1)[1]
#     X_pgd_logit = model(X_pgd)
#     X_pgd_plabel = torch.max(X_pgd_logit, dim=1)[1]
#     rob = (X_pgd_plabel == y).float().sum()
#     acc = (X_plabel == y).float().sum()
#     adv = X_pgd.cpu().detach().numpy()
#     eta = (X_pgd - X).cpu().detach().numpy()
#     return acc.item(), rob.item(), adv, eta


def eval_adv_test_whitebox(model, device, test_loader,
                           epsilon, num_steps, step_size, args):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust = 0
    accurate = 0
    total = 0
    test_advs = []
    test_etas = []

    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        acc, rob, adv, eta, _ = _pgd(
            model, X, y, epsilon, num_steps, step_size, args.random, device
            )
        # print("Batch {}/{}, Accuracy: {:.2f}%\tRobustness: {:.2f}%".format(
        #     i, len(test_loader), acc / len(y) * 100, rob / len(y) * 100),
        #     end="\r"
        #     )
        robust += rob
        accurate += acc
        total += target.shape[0]
        test_advs.append(adv)
        test_etas.append(eta)
    # print()
    if args.save_attack:
        te_adv = np.concatenate(test_advs, 0)
        te_eta = np.concatenate(test_etas, 0)
    else:
        te_adv = te_eta = None
    return accurate / total, robust / total, te_adv, te_eta


def set_default_hyper(args):
    keys = ["test_epsilon", "test_num_steps", "test_step_size", "net"]
    if args.mnist:
        values = [0.3, 40, 0.01, "nn"]
    elif args.cifar and args.pgd10:
        values = [0.031, 10, 0.007, "resnet"]
    elif args.cifar and args.pgd20:
        values = [0.031, 20, 0.003, "resnet"]
    else:
        raise ValueError("How come you reach here?")

    for key, value in zip(keys, values):
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def check_args(args):
    if (args.mnist and args.cifar) is True:
        raise ValueError("Cannot use --mnist and --cifar at the same time.")

    if (args.mnist or args.cifar) is False:
        raise ValueError("Specify one of --mnist and --cifar.")

    if (args.pgd10 and args.pgd20) is True:
        raise ValueError("Cannot use --pgd10 and --pgd20 at the same time.")

    if (args.pgd10 or args.pgd20) is False:
        raise ValueError("Specify one of --pgd10 and --pgd20.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true", default=False)
    parser.add_argument("--cifar", action="store_true", default=False)
    parser.add_argument("--net", type=str,
                        choices=["wideres", "resnet", "nn"])
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=100, metavar='N')
    parser.add_argument('--test-epsilon', type=float,
                        help='perturbation')
    parser.add_argument('--test-num-steps', type=int,
                        help='perturb number of steps')
    parser.add_argument('--test-step-size', type=float,
                        help='perturb step size')
    parser.add_argument("--pgd10", action="store_true")
    parser.add_argument("--pgd20", action="store_true")
    parser.add_argument("--train-mode-eval", action="store_true")
    parser.add_argument("--load-data", type=str, default=None)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--random', default=True,
                        help='random initialization for PGD')
    parser.add_argument("--seed", type=int, default=0,
                        help="set to -1 to use random without a seed")
    parser.add_argument("--data-root", default=".", type=str)
    parser.add_argument("--save-attack", default=False)
    args = parser.parse_args()

    check_args(args)
    set_default_hyper(args)
    device, kwargs = set_device(args)
    model = set_network(args, device)
    model.load_state_dict(torch.load(args.load))
    train_loader, test_loader = load_dataset(args)

    te_acc, te_rob, _, _ = eval_adv_test_whitebox(
        model, device, train_loader, args.test_epsilon, args.test_num_steps,
        args.test_step_size, args
        )

    print("Test Accuracy: {:.2f}%\tTest Robustness: {:.2f}%".format(
        te_acc * 100, te_rob * 100
        ))