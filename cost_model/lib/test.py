import os
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import torch
from torch.utils.data import DataLoader

from utils import *
from preprocess import preprocess
from dataset import MTDataset
from model import LinReg, dnn

if __name__ == "__main__":
    args = set_arg()
    
    # Fix randon seed for reproducibility
    same_seeds(args.seed)
    os.chdir(args.home_dir)
    '''
    Data Preprocess
    '''
    test_data_dir = args.data_dir / "testing_data/fio_known"
    #test_data_dir = args.data_dir / "tmp"
    counter_path = args.data_dir / "counters.txt"
    cache_dir = Path("cache")
    
    # If preprocessed data exists in cache folder, load it directly.
    IS_CACHED = True

    # Create a cache directory
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
        IS_CACHED = False
    
    # Read data
    data_cpath = cache_dir / "test_data.pickle"
    IS_CACHED = IS_CACHED and os.path.exists(data_cpath)
    if IS_CACHED:
        with data_cpath.open("rb") as f:
            data = pickle.load(f)
    else:
        data = preprocess(args, test_data_dir, counter_path)
        with data_cpath.open("wb") as f:
            pickle.dump(data, f)

    instances, labels = data["instances"], data["labels"]
    print("Size of raw testing data: {}".format(instances.shape))

    if args.model_type == "LinReg":
        model_path = args.param_dir / "LinReg.pickle"
        with model_path.open("rb") as f:
            clf = pickle.load(f)

        LinReg.test(clf, instances, labels)
    elif args.model_type == "dnn":
        raise "1"
    else:
        raise "No this model supported!"