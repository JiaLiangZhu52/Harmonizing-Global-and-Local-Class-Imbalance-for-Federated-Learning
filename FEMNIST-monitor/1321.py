import torchvision
import matplotlib
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
from utils.sampling import EMNIST_client_imbalance, load_EMNIST_data, EMNIST_client_regenerate, ratio_loss_data,make_transforms,get_auxiliary_data,load_dataset
from utils.options import args_parser
from models.Update import LocalUpdate,compute_pr,DatasetSplit,Dataset
from models.Nets import MLP, CNNMnist, CNNCifar, Net
from models.Fed import FedAvg, outlier_detect, whole_determination, monitoring, ground_truth_composition, cosine_similarity
from models.test import test_img


x=np.arange(0,60,10)
print(x)