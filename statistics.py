import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def reduce_max(array, axises):
    res = array
    axises = reversed(sorted(axises))
    for a in axises:
        res = np.max(res, axis=a)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--eta", action="store_true", default=True)
    parser.add_argument("--save-folder", type=str)
    args = parser.parse_args()

    files = sorted(os.listdir(args.folder))
    all_eta = []
    all_adv = []
    if args.eta:
        for file in files:
            if file.startswith("eta"):
                eta = np.load(os.path.join(args.folder, file))
                plt.hist(eta.flatten(), 20)
                plt.ylim(0, 1.5e7)
                plt.savefig(os.path.join(args.save_folder, file[:-4]))
                plt.clf()
