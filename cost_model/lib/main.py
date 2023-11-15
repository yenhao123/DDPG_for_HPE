import os
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import *
from preprocess import preprocess
from dataset import MTDataset
from model import ridge, dnn


if __name__ == "__main__":
    args = set_arg()
    
    # Fix randon seed for reproducibility
    same_seeds(args.seed)
    os.chdir(args.home_dir)

    '''
    Data Preprocess
    '''
    train_data_dir = args.data_dir / "train_data"
    train_data_dir = args.data_dir / "testing_data/fio_known"
    counter_path = args.data_dir / "counters.txt"
    cache_dir = Path("cache")
    param_dir = Path("parm")

    # If preprocessed data exists in cache folder, load it directly.
    IS_CACHED = True

    # Create a cache directory
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
        IS_CACHED = False

    # Read data
    data_cpath = cache_dir / "data.pickle"
    IS_CACHED = IS_CACHED and os.path.exists(data_cpath)
    if IS_CACHED:
        with data_cpath.open("rb") as f:
            data = pickle.load(f)
    else:
        data = preprocess(args, train_data_dir, counter_path)
        with data_cpath.open("wb") as f:
            pickle.dump(data, f)

    instances, labels = data["instances"], data["labels"]

    VAL_RATIO = 0.2
    x_train, x_val, y_train, y_val = train_test_split(instances, labels, test_size=VAL_RATIO, random_state=args.seed)

    print("Size of train set: {}".format(x_train.shape))
    print("Size of validation set: {}".format(x_val.shape))

    train_set = MTDataset(x_train, y_train)
    val_set = MTDataset(x_val, y_val)
    
    '''
    Training
    '''
    if args.model_type == "ridge":
        clf = ridge.train(x_train, y_train)
        ridge.test(clf, x_train, y_train)
    elif args.model_type == "dnn":
        # Build data loader from the dataset
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        dnn.train(args, train_loader, val_loader)
