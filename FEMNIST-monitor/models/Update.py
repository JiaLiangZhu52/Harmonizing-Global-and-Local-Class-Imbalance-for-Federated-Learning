import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.preprocessing import label_binarize

class RatioLossFL(nn.Module):  # GL-loss
    def __init__(self, args, class_num, alpha=1, beta=.1, size_average=True, pr=None, Hl=None):
        super(RatioLossFL, self).__init__()
        self.args = args
        self.alpha = alpha
        self.beta = beta
        self.class_num = class_num
        self.size_average = size_average
        self.pr = pr
        self.Hl = Hl

    def forward(self, inputs, targets, pr, Hl):
        self.alpha = torch.ones(self.class_num)
        self.beta = torch.ones(self.class_num) * 0.1
        self.beta = self.beta.cuda()
        self.alpha = self.alpha.cuda()
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        log_p = P.log()  # 64X26
        log_p = log_p.to(self.args.device)
        loss_weights = (self.alpha + self.beta * pr) * (self.alpha + self.beta * Hl)  # 26x1

        loss_fun = torch.nn.CrossEntropyLoss(weight=loss_weights)
        batch_loss = loss_fun(log_p, targets)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def compute_Hl(net_local, self):
    idxs_aux = []

    number_raw1 = torch.zeros(self.args.num_classes, dtype=torch.float32)
    for idxs_user in range(self.num_class):
        number_raw1[idxs_user] = len(self.dict_users[idxs_user])

    for i in range(self.num_class):
        if number_raw1[i] == 0:
            idxs_aux.append(i)

    number_raw1 = torch.mean(number_raw1) / number_raw1 * 10

    for i in range(len(idxs_aux)):
        number_raw1[idxs_aux[i]] = 1

    Hl = []
    for i in range(self.num_class):
        Hl.append(number_raw1[i])
    Hl = torch.tensor(Hl)
    Hl = Hl.to(self.args.device)
    return Hl


def compute_pr(args, model, dataset, label, dict_ratio):
    model.requires_grad_(True)
    n_classes = len(dict_ratio)
    loss_fn = torch.nn.CrossEntropyLoss()
    Delta_W = []
    for idxs in range(n_classes):
        ldr_train_aux = DataLoader(DatasetSplit(dataset, label, dict_ratio[idxs]), args.local_bs, shuffle=True)
        for batch_idx, (images, labels) in enumerate(ldr_train_aux):
            if args.dataset == 'femnist' or 'mnist':
                images = images.unsqueeze(1)
            elif args.dataset == 'cifar10':
                images = images.swapaxes(1, 3).swapaxes(2, 3)
            elif args.dataset == 'cifar100':
                images = images.swapaxes(1, 3).swapaxes(2, 3)

            images, labels = images.to(args.device, dtype=torch.float), labels.to(args.device,
                                                                                  dtype=torch.long)
            model.zero_grad()
            f_data = model(images)
            loss = loss_fn(f_data, labels)
            grad_c = torch.autograd.grad(loss, model.parameters())
            Delta_W.append(grad_c[-2])
    Delta_W_sum = torch.sum(torch.stack(Delta_W, dim=0), dim=0)
    pr = [torch.abs(torch.mean(
        (n_classes - 1) * Delta_W[p] / (Delta_W_sum - Delta_W[p]),
        dim=1)[p]) for p in range(n_classes)]
    pr = torch.stack(pr)
    # Ra_p=torch.softmax(Ra_p,dim=0)
    pr = pr / torch.sum(pr) * 100

    model.requires_grad_(False)
    return pr


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """


    def __init__(self, args, class_num, alpha=None, gamma=2, size_average=True):
        self.args = args
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        alpha = alpha.to(self.args.device)
        probs = probs.to(self.args.device)
        log_p = log_p.to(self.args.device)

        batch_loss = -alpha * torch.pow((1 - probs), self.gamma) * log_p  # torch.pow求次方的函数
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class Ratio_Cross_Entropy(nn.Module):  # ratio Loss
    def __init__(self, args, class_num, alpha=1, beta=.1, size_average=True, pr=None):
        super(Ratio_Cross_Entropy, self).__init__()
        self.args = args
        self.alpha = alpha
        self.beta = beta
        self.class_num = class_num
        self.size_average = size_average
        self.pr = pr

    def forward(self, inputs, targets, pr):
        self.alpha = torch.ones(self.class_num)
        self.beta = torch.ones(self.class_num) * 0.1
        self.beta = self.beta.cuda()
        self.alpha = self.alpha.cuda()

        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)


        log_p = P.log()
        log_p = log_p.to(self.args.device)
        loss_weights = self.alpha + self.beta * pr
        loss_fun = torch.nn.CrossEntropyLoss(weight=loss_weights)  # ratio loss
        # loss_fun = torch.nn.CrossEntropyLoss()  #ce loss

        batch_loss = loss_fun(log_p, targets)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss









class DatasetSplit(Dataset):
    def __init__(self, dataset, labels, idxs):
        self.dataset = dataset
        self.labels = labels
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if len(self.labels) != 0:
            image = self.dataset[self.idxs[item]]
            label = self.labels[self.idxs[item]]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label


def AccuarcyCompute(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()

    test_np = np.argmax(pred, 1)
    count = 0
    for i in range(len(test_np)):
        if test_np[i] == label[i]:
            count += 1
    return np.sum(count), len(test_np)


class LocalUpdate(object):
    def __init__(self, args, dataset=None, label=None, idxs=None, alpha=None, beta=None, pr=None, size_average=False,
                 is_prhl=True):
        self.pr = pr
        self.args = args
        self.is_prhl = is_prhl
        if self.args.loss == 'mse':
            self.loss_func = nn.MSELoss(reduction='mean')
        elif self.args.loss == 'focal':
            self.loss_func = FocalLoss(class_num=args.num_classes, alpha=alpha, args=args, size_average=size_average)
        elif self.args.loss == 'ratio' or self.args.loss == 'ce':
            self.loss_func = Ratio_Cross_Entropy(class_num=args.num_classes, alpha=alpha, beta=beta,
                                                 size_average=size_average, args=args)


        elif self.args.loss == 'ratiofl': #GL Loss
            self.loss_func = RatioLossFL(class_num=args.num_classes, alpha=alpha, beta=beta, size_average=size_average,
                                         args=args)

        self.loss_func_aux = Ratio_Cross_Entropy(class_num=args.num_classes, alpha=alpha, beta=beta,
                                                 size_average=size_average, args=args)

        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, label, idxs), batch_size=self.args.local_bs,
                                    shuffle=True)

        self.num_class = args.num_classes

        dict_users = {i: np.array([], dtype='int64') for i in range(self.num_class)}
        for class_index in range(self.num_class):
            for batch_idx, (images_local, labels_local) in enumerate(self.ldr_train):
                idx_temp = np.where(labels_local == class_index)
                dict_users[class_index] = np.concatenate((dict_users[class_index], idx_temp[0][:]), axis=0)
        self.dict_users = dict_users
        self.dataset = dataset
        self.label = label

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=5e-4)

        epoch_loss = []
        epoch_ac = []

        ########################## compute Hl
        if self.args.loss == 'ratiofl' and self.is_prhl == True:
            net_local = net
            Hl = compute_Hl(net_local, self)
        ################################
        for iter in range(self.args.local_ep):
            batch_loss = []
            batch_ac = []
            batch_whole = []
            for batch_idx, (images, labels) in enumerate(
                    self.ldr_train):
                if self.args.dataset == 'femnist' or 'mnist':
                    images = images.unsqueeze(
                        1)

                elif self.args.dataset == 'cifar10':
                    images = images.swapaxes(1, 3).swapaxes(2, 3)
                net.requires_grad_(True)
                images, labels = images.to(self.args.device, dtype=torch.float), labels.to(self.args.device,
                                                                                           dtype=torch.long)
                net.zero_grad()

                log_probs = net(images)
                ac, whole = AccuarcyCompute(log_probs, labels)


                if self.args.loss == 'ratio' or self.is_prhl==False :
                    loss = self.loss_func_aux(log_probs, labels, self.pr)

                elif self.args.loss == 'ratiofl' :
                    loss = self.loss_func(log_probs, labels, self.pr, Hl)

               # loss = self.loss_func(log_probs, labels)    #  focal loss

                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(iter, batch_idx * len(images),
                                                                                    len(self.ldr_train.dataset),
                                                                                    100. * batch_idx / len(
                                                                                        self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
                batch_ac.append(ac)
                batch_whole.append(whole)
            epoch_ac.append(sum(batch_ac) / sum(batch_whole))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_ac) / len(epoch_ac)
