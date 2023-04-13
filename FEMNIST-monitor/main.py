import matplotlib
import torchvision

matplotlib.use('Agg')
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from PIL import Image
import math
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
from utils.sampling import EMNIST_client_imbalance, load_EMNIST_data, EMNIST_client_regenerate,EMNIST_client_imbalance_cifar10, ratio_loss_data,make_transforms,get_auxiliary_data,load_dataset
from utils.options import args_parser
from models.Update import LocalUpdate,compute_pr,DatasetSplit,Dataset
from models.Nets import MLP, CNNMnist, CNNCifar, Net
from models.Fed import FedAvg, ground_truth_composition
from models.test import test_img
if __name__ == '__main__':
    # parse args
    args = args_parser()
    print("args.gpu=",args.gpu)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # load dataset and split users
    if args.dataset == 'femnist':

        dataset_train, label_train, dataset_test, label_test, w_train, w_test = load_EMNIST_data('../emnist-letters.mat', verbose=True, standarized=False)

        if args.iid != 0:
            dict_users = EMNIST_client_imbalance(args,dataset_train, label_train, w_train, 100, [0, 1, 2,3,4,5,6], 0.1)
        else:
            dict_users = EMNIST_client_regenerate(dataset_train, label_train, w_train, 100)

#########################################
    elif args.dataset == 'cifar10':
        dataset_train1, dataset_test1, n_classes, n_channels, img_size= load_dataset( "cifar10")
        args.num_channels=n_channels
        args.num_classes=n_classes
        w_train=np.random.randint(0,100,50000)

        dataset_train=dataset_train1.data
        label_train=dataset_train1.targets
        dataset_test=dataset_test1.data
        label_test=dataset_test1.targets


        if args.iid != 0:
            dict_users = EMNIST_client_imbalance_cifar10(args,dataset_train, label_train, w_train, 100, [0, 1, 2], 0.1)
        else:
            dict_users = EMNIST_client_regenerate(dataset_train, label_train, w_train, 100)

    elif args.dataset == 'mnist':
        dataset_train1, dataset_test1, n_classes, n_channels, img_size = load_dataset("mnist")
        args.num_channels = n_channels
        args.num_classes = n_classes
        w_train = np.random.randint(0, 100, 60000)

        dataset_train = dataset_train1.data
        label_train = dataset_train1.targets
        dataset_test = dataset_test1.data
        label_test = dataset_test1.targets

        if args.iid != 0:
            dict_users = EMNIST_client_imbalance_cifar10(args, dataset_train, label_train, w_train, 100, [0, 1, 2],
                                                         0.02)
        else:
            dict_users = EMNIST_client_regenerate(dataset_train, label_train, w_train, 100)
###########################################
    else:
        exit('Error: unrecognized dataset')

        # build model
    if args.model == 'cnn' and args.dataset == 'femnist':
        net_glob = Net().to(args.device)
    elif args.model == 'cnn' and args.dataset == 'cifar10':
        net_glob = Net().to(args.device)

    elif args.model == 'cnn' and args.dataset == 'mnist':

        net_glob = Net().to(args.device)

    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    sim_1 = []
    selection = []
    labels = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    ratio = None
    val_acc_list, net_list = [], []
    pr=torch.ones([args.num_classes])
    pr = pr.to(args.device)
    dict_ratio = ratio_loss_data(dataset_train, label_train, w_train, args.num_classes, args)

    for iter in range(args.epochs):

        if iter > 135:
            args.lr = 0.01
        w_locals, loss_locals, ac_locals, num_samples = [], [], [], []
        net_glob_aux=net_glob
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)


        pro_ground_truth = ground_truth_composition(dict_users, idxs_users, args.num_classes, label_train)
        print("pro_ground_truth=",pro_ground_truth)

        #Print the number of samples for each selected customer
        for i in idxs_users:
            number_raw1 = np.zeros((1, 26))
            for label in range(26):
                temp = 0
                for index in dict_users[i]:
                    if label_train[index] == label:
                        temp += 1
                number_raw1[0, label] += temp

            print("被选中的第",i,"个客户", number_raw1)

####################


        pr = compute_pr(args, net_glob_aux, dataset_train, label_train, dict_ratio)
        print("pr=", pr)

        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, label=label_train, idxs=dict_users[idx], alpha=None,beta=None, pr=pr, size_average=True,is_prhl=True) #ratiofl
            w, loss, ac= local.train(net=copy.deepcopy(net_glob).to(args.device))


            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            ac_locals.append(copy.deepcopy(ac))
            num_samples.append(len(dict_users[idx]))

        # monitor
        if args.loss == 'ratio' or args.loss == 'ratiofl' or args.loss == 'focal':
            cc_net, cc_loss = [], []
            aux_class = [i for i in range(args.num_classes)]

            for i in aux_class:
                cc_local = LocalUpdate(args=args, dataset=dataset_train, label=label_train, idxs=dict_ratio[i], alpha=None,beta=None,pr=pr, size_average=True,is_prhl=False)

                cc_w, cc_lo, cc_ac = cc_local.train(net=copy.deepcopy(net_glob).to(args.device))
                cc_net.append(copy.deepcopy(cc_w))
                cc_loss.append(copy.deepcopy(cc_lo))

        w_glob_last = copy.deepcopy(w_glob)
        w_glob = FedAvg(w_locals)

        num_sample = np.sum(num_samples)

        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        ac_avg = sum(ac_locals) / len(ac_locals)
        print('Round {:3d}, Average loss {:.3f}, Accuracy {:.3f}\n'.format(iter, loss_avg, ac_avg))
        loss_train.append(loss_avg)

    # testing

        print("iter==",iter)
        np.savetxt('{}-num-{}.csv'.format(args.dataset, int(args.frac * args.num_users)), sim_1, delimiter=',')

        net_glob.eval()
        print('FL(femnist, mismatch 4): {}, 10:1, [0, 1, 2,3,4,5,6], {} eps, {} local eps'.format(args.loss, args.epochs, args.local_ep))

        acc_test, loss_test = test_img(net_glob, dataset_test, label_test, args)
        print("Testing accuracy: {:.2f}".format(acc_test))
        print("Testing loss: {:.2f}".format(loss_test))
