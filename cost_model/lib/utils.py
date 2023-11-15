import argparse
import pathlib
import torch
import numpy as np


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home_dir", type=pathlib.Path, required=True)
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    
    # Data preprocess
    parser.add_argument("--n_counters", type=int, default=103)
    parser.add_argument("--seed", type=int, default=0)
    
    # Training/Testing
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epoch", type=int, default=20)
    return parser.parse_args()