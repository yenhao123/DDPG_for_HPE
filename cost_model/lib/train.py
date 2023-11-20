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

def plot_learning_curve(args, record, title='', ylabel='', img=''):
    ''' Plot learning curve of DNN (train & validation) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1
    figure(figsize=(6, 4))
    plt.plot(x_1, record['train'], c='tab:red', label='Train', alpha=0.6)
    if img == "loss":
        plt.ylim([0, record['train'][1]])
    elif img == "acc":
        plt.ylim([0, 100])
    plt.plot(x_2, record['val'], c='tab:cyan', label='Validation', alpha=0.6)
    plt.xlabel('Training Steps')
    plt.ylabel(ylabel)
    plt.title('Learning Curve of {}'.format(title))
    plt.legend()
    plt.savefig(args.cache / (img+".jpg"))

if __name__ == "__main__":
    args = set_arg()
    
    # Fix randon seed for reproducibility
    same_seeds(args.seed)
    os.chdir(args.home_dir)

    '''
    Data Preprocess
    '''
    train_data_dir = args.data_dir / "train_data"
    counter_path = args.data_dir / "counters.txt"
    cache_dir = Path("cache")

    # If preprocessed data exists in cache folder, load it directly.
    IS_CACHED = False

    # Create a cache directory
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
        IS_CACHED = False

    # Read data
    data_cpath = cache_dir / ("train_data_ws=" + str(args.window_size) + ".pickle")
    IS_CACHED = IS_CACHED and os.path.exists(data_cpath)
    if IS_CACHED:
        with data_cpath.open("rb") as f:
            data = pickle.load(f)
    else:
        data = preprocess(args, train_data_dir, counter_path)
        with data_cpath.open("wb") as f:
            pickle.dump(data, f)

    instances, labels = data["instances"], data["labels"]
    print("Size of raw training data: {}".format(instances.shape))
    '''
    Training/Validating
    '''
    if args.model_type == "LinReg":
        clf = LinReg.train(instances, labels)
        model_o_path = args.param_dir / "LinReg.pickle"
        with model_o_path.open("wb") as f:
            pickle.dump(clf, f)
    elif args.model_type == "dnn":
        VAL_RATIO = 0.2
        x_train, x_val, y_train, y_val = train_test_split(instances, labels, test_size=VAL_RATIO, random_state=args.seed)

        print("Size of train set: {}".format(x_train.shape))
        print("Size of validation set: {}".format(x_val.shape))
        train_set = MTDataset(x_train, y_train)
        val_set = MTDataset(x_val, y_val)
        # Build data loader from the dataset
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

        acc_record, loss_record = dnn.train(args, train_loader, val_loader)

        plot_learning_curve(args, loss_record, title='Loss', ylabel='Mean Squared Error', img='loss')

        acc_record_percent = {
            'train': list(map(lambda x: x * 100, acc_record['train'])),
            'val': list(map(lambda x: x * 100, acc_record['val'])),
        }
        plot_learning_curve(acc_record_percent, title='Accuracy', ylabel='Accuracy (%)', img='acc')
    else:
        raise "No this model supported!"