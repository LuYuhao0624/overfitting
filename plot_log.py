import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_value(line, patern1, patern2="[0-9]+.[0-9]+"):
    term = re.search(patern1, line)
    value = re.search(patern2, term.group(0))
    value = float(value.group(0))
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()

    with open(args.log, "r") as f:
        lines = f.readlines()

    tr_rob = []
    tr_sta = []
    te_acc = []
    pgd_10 = []
    pgd_20 = []
    x_rate = [(i + 1) for i in range(100)]
    l_as = []
    l_ss = []
    x_loss = [0.2 * (i + 1) for i in range(100 * args.interval)]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for line in lines:
        if line[0] == "T":
            tr_rob_rate = get_value(line, "Train robustness: [0-9.]*")
            tr_sta_rate = get_value(line, "stability: [0-9.]*")
            te_acc_rate = get_value(line, "Test accuracy: [0-9.]*")
            pgd_10_rate = get_value(line, "PGD-10: [0-9.]*")
            pgd_20_rate = get_value(line, "PGD-20: [0-9.]*")
            tr_rob.append(tr_rob_rate)
            tr_sta.append(tr_sta_rate)
            te_acc.append(te_acc_rate)
            pgd_10.append(pgd_10_rate)
            pgd_20.append(pgd_20_rate)
        if line[0] == "[":
            l_a = get_value(line, "A: [0-9.]*")
            l_s = get_value(line, "S: [0-9.]*")
            l_as.append(l_a)
            l_ss.append(l_s)

    metrics = [tr_rob, tr_sta, te_acc, pgd_10,
               # pgd_20
               ]
    labels = ["train robustness", "train stability", "test accuracy",
              "PGD-10", "PGD-20"]

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    for i in range(len(metrics)):
        ax1.plot(x_rate, metrics[i], label=labels[i],
                 color=colors[i], linestyle="-")
    ax1.set_ylabel('rate')

    # ax2 = ax1.twinx()  # this is the important function
    # ax2.plot(x_loss, l_as, "r", label=r"$L_a$", linestyle="--")
    # ax2.plot(x_loss, l_ss, "b", label=r"$L_s$", linestyle="--")
    # ax2.set_xlim([0, 100])
    # ax2.set_ylabel("loss")

    fig.legend(loc=1, bbox_to_anchor=(1, 0.95), bbox_transform=ax1.transAxes)

    plt.show()