import torch
import argparse
from utils import *
from evaluate import _pgd
from torch.autograd import Variable


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist", action="store_true")
    parser.add_argument("--cifar", action="store_true")
    parser.add_argument("--net", type=str, choices=["wideres", "resnet", "nn"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--train-epsilon", type=float, default=None)
    parser.add_argument("--train-num-steps", type=int, default=None)
    parser.add_argument("--train-step-size", type=float, default=None)
    parser.add_argument("--test-epsilon", type=float, default=None)
    parser.add_argument("--test-num-steps", type=int, default=None)
    parser.add_argument("--test-step-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=None)
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--data-root", default="~/data", type=str)
    parser.add_argument("--load-data", type=str, default=None)
    args = parser.parse_args()

    device, _ = set_device(args)
    model = set_network(args, device)
    set_hyper(args)
    model.load_state_dict(torch.load(args.load))
    train_loader, test_loader = load_dataset(args)

    model.eval()
    back_label = 0
    back_plabel = 0
    fwd_mid_rob = 0
    bwd_mid_rob = 0
    total = 0

    for i, (data, label) in enumerate(test_loader):
        data, label = data.to(device), label.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(label)
        _, _, _, _, adv = _pgd(
            model, X, y, args.train_epsilon, args.train_num_steps,
            args.train_step_size, True, device
            )
        adv_logit = model(adv)
        adv_plabel = torch.max(adv_logit, dim=1)[1]

        _, _, _, _, double_adv = _pgd(
            model, X, adv_plabel, args.test_epsilon, args.test_num_steps,
            args.test_step_size, False, device
            )
        double_adv_logit = model(double_adv)
        double_adv_plabel = torch.max(double_adv_logit, dim=1)[1]
        clean_logit = model(X)
        clean_plabel = torch.max(clean_logit, dim=1)[1]
        back_label += (double_adv_plabel == label).float().sum()
        back_plabel += (double_adv_plabel == clean_plabel).float().sum()
        total += len(label)

        fwd_mid = (X + adv) / 2
        fwd_mid_logit = model(fwd_mid)
        fwd_mid_plabel = torch.max(fwd_mid_logit, dim=1)[1]
        bwd_mid = (adv + double_adv) / 2
        bwd_mid_logit = model(bwd_mid)
        bwd_mid_plabel = torch.max(bwd_mid_logit, dim=1)[1]

        fwd_mid_rob += (fwd_mid_plabel == label).float().sum()
        bwd_mid_rob += (bwd_mid_plabel == label).float().sum()

        print("{}/{} Back rate: {:.2f}% (label), {:.2f}% (plabel); "
              "Robustness: Fwd mid {:.2f}%, Bwd mid {:.2f}%".format(
            i + 1, len(test_loader),
            100 * back_label / total, 100 * back_plabel / total,
            100 * fwd_mid_rob / total, 100 * bwd_mid_rob / total
            ), end="\r")
    print()
    print("Final back rate: {:.2f}% (label), {:.2f}% (plabel); "
          "Robustness: Fwd mid {:.2f}%, Bwd mid {:.2f}%".format(
        100 * back_label / total, 100 * back_plabel / total,
        100 * fwd_mid_rob / total, 100 * bwd_mid_rob / total
        ))
