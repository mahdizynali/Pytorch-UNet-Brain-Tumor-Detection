import sys, os, glob, random, time
import numpy as np
import pandas as pd
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
plt.style.use("dark_background")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import torchvision.transforms as tt

import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# data preprocessing
file_path = os.path.dirname(__file__) + "/"
dataSets = file_path + "dataSets/*"

num_epochs = 30 # number of training duration
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64


# set a seeds to use for some probable randomly works

def set_seed(seed = 0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
set_seed()
