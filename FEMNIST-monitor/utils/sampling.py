import numpy as np
import scipy.io as sio
import torch
import  torchvision
from torchvision import datasets, transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def load_dataset(dataset):
    if dataset == "cifar10":
        dataset_train =torchvision.datasets.CIFAR10(root="./cifar10",transform=torchvision.transforms.ToTensor(),train=True, download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = np.array(dataset_train.targets)
        dataset_test = datasets.CIFAR10(root="./cifar10", train=False,transform=torchvision.transforms.ToTensor(),download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 10
        n_channels = 3
        img_size = 32
    elif dataset == "cifar100":
        dataset_train = torchvision.datasets.CIFAR100(root="./cifar100",transform=torchvision.transforms.ToTensor(),train=True, download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets =np.array(dataset_train.targets)
        dataset_test = datasets.CIFAR100(root="./cifar100", train=False,transform=torchvision.transforms.ToTensor(),download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 100
        n_channels = 3
        img_size = 32
    elif dataset == "mnist":
        dataset_train = datasets.MNIST(root="./mnist",transform=torchvision.transforms.ToTensor(),train=True, download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets =np.array(dataset_train.targets)
        dataset_test = datasets.MNIST(root="./mnist", train=False,transform=torchvision.transforms.ToTensor(),download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 10
        n_channels = 1
        img_size = 28
    elif dataset == "fashion-mnist":
        dataset_train = datasets.FashionMNIST(root='datasets/' + dataset, download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets))
        dataset_test = datasets.FashionMNIST(root='datasets/' + dataset, train=False, download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 10
        n_channels = 1
        img_size = 28
    elif dataset == "emnist-letter":
        dataset_train = datasets.EMNIST(root='datasets/' + "emnist", split="letters", download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets)) - 1
        dataset_test = datasets.EMNIST(root='datasets/' + "emnist", split="letters", train=False, download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets)) - 1
        n_classes = 26
        n_channels = 1
        img_size = 28
    elif dataset == "emnist-digit":
        dataset_train = datasets.EMNIST(root='datasets/' + "emnist", split="digits", download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets))
        dataset_test = datasets.EMNIST(root='datasets/' + "emnist", split="digits", train=False, download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 10
        n_channels = 1
        img_size = 28
    else:
        raise NotImplementedError

    return dataset_train, dataset_test, n_classes, n_channels, img_size



def load_EMNIST_data(file, verbose = False, standarized = False):
    """
    file should be the downloaded EMNIST file in .mat format.
    """
    mat = sio.loadmat(file)
    data = mat["dataset"]



    writer_ids_train = data['train'][0,0]['writers'][0,0]

   # print(" writer_ids_train===========", writer_ids_train)

    writer_ids_train = np.squeeze(writer_ids_train)
    X_train = data['train'][0,0]['images'][0,0]
    X_train = X_train.reshape((X_train.shape[0], 28, 28), order = "F")
    y_train = data['train'][0,0]['labels'][0,0]
    y_train = np.squeeze(y_train)
    y_train -= 1 #y_train is zero-based

    writer_ids_test = data['test'][0,0]['writers'][0,0]
    writer_ids_test = np.squeeze(writer_ids_test)
    X_test = data['test'][0,0]['images'][0,0]
    X_test= X_test.reshape((X_test.shape[0], 28, 28), order = "F")
    y_test = data['test'][0,0]['labels'][0,0]
    y_test = np.squeeze(y_test)
    y_test -= 1 #y_test is zero-based


    if standarized:
        X_train = X_train/255
        X_test = X_test/255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image


    if verbose == True:
        print("EMNIST-letter dataset ... ")
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)

    return X_train, y_train, X_test, y_test, writer_ids_train, writer_ids_test


def EMNIST_client_regenerate(data_train, label_train, writer_train, num_users):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(num_users):
        temp = np.where(writer_train == i)
        dict_users[i] = np.concatenate((dict_users[i], temp[0][:]), axis=0)
    return dict_users


    return dict_users

def ratio_loss_data(data_train, label_train, writer_train, num_class, args):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_class)}

    for class_index in range(num_class):
        idx_temp = np.where(label_train == class_index)
        dict_users[class_index] = np.concatenate((dict_users[class_index], idx_temp[0][0:args.local_bs]), axis=0)
    return dict_users





def EMNIST_client_imbalance(args,data_train, label_train, writer_train, num_users, minor_class, ratio):

    dict_raw = EMNIST_client_regenerate(data_train, label_train, writer_train, num_users)
    dict_users = dict_raw

    ####################
    number_raw = np.zeros((1,  args.num_classes))
    for i in range(num_users):
        for label in range( args.num_classes):
            temp = 0
            for index in dict_raw[i]:
                if label_train[index] == label:
                    temp += 1
            number_raw[0, label] += temp
    for minor in minor_class:
        base_temp = int(number_raw[0, minor] * ratio)
        raw_temp = number_raw[0, minor]

        for i in range(num_users):
            for index in dict_raw[i]:
                if label_train[index] == minor:
                    dict_users[i] = np.delete(dict_users[i], np.where(dict_users[i] == index)[0])
                    raw_temp -= 1
                if raw_temp == base_temp:
                    break


    return dict_users

def EMNIST_client_imbalance_cifar10(args,data_train, label_train, writer_train, num_users, minor_class, ratio):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for i in range(num_users):
        temp = np.where(writer_train == i)
        dict_users[i] = np.concatenate((dict_users[i], temp[0][:]), axis=0)


    dict_raw = dict_users

    ####################
    number_raw = np.zeros((1,  args.num_classes))
    for i in range(num_users):
        for label in range( args.num_classes):
            temp = 0
            for index in dict_raw[i]:
                if label_train[index] == label:
                    temp += 1
            number_raw[0, label] += temp
    for minor in minor_class:
        base_temp = int(number_raw[0, minor] * ratio)
        raw_temp = number_raw[0, minor]

        for i in range(num_users):
            for index in dict_raw[i]:
                if label_train[index] == minor:
                    dict_users[i] = np.delete(dict_users[i], np.where(dict_users[i] == index)[0])
                    raw_temp -= 1
                if raw_temp == base_temp:
                    break


    return dict_users



def get_auxiliary_data(data_train, label_train, num_class, args):



    dict_users = []

    for class_index in range(num_class):
        idx_temp = np.where(label_train == class_index)
        dict_users[class_index] = np.concatenate((dict_users[class_index], idx_temp[0][0:args.local_bs]), axis=0)
    return dict_users




normalize_cifar10 = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
normalize_mnist = transforms.Normalize(mean=(0.1307,), std=(0.3081,))

def make_transforms(args, dataset, train=True):
    if dataset == "cifar10" or dataset == "cifar100":
        if train:
            if not args.no_data_augmentation:
                transform = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    normalize_cifar10,
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize_cifar10,
                ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize_cifar10,
            ])
    elif dataset == "mnist":
        if train:
            if not args.no_data_augmentation:
                transform = transforms.Compose([
                    # transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    normalize_mnist,
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize_mnist,
                ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize_mnist,
            ])
    elif dataset == "fashion-mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif dataset == "emnist-letter" or "emnist-digit":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        raise NotImplementedError

    return transform