import copy
import torch
import numpy as np
from scipy.linalg import solve


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def ground_truth_composition(dict_users, idxs_users, num_class, labels):
    res = [0 for i in range(num_class)]
    for idx in idxs_users:
        for i in dict_users[idx]:
            for j in range(num_class):
                temp = np.where(labels[i] == j)
                res[j] += len(temp[0])
    return res
