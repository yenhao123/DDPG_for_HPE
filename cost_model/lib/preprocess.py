import numpy as np
import pandas as pd
import re

# Get extracted counters
def get_counter(counter_path):
    if not counter_path.exists:
        raise Exception("counter is not found")

    with counter_path.open("r") as f:
        lines = f.readlines()

    return [line.strip() for line in lines]

# Rename counters, e.g., process(fio) -> process(app)
def rename_counter(df):
    rename_list = {}
    for col_name in df.columns:
        new_col_name = re.sub("^Process\([\w#]+\)", "Process(app)", col_name)
        rename_list.update({col_name: new_col_name})

    return df.rename(columns=rename_list)

# Extract samples within the effective duration
def filter_rows(df):
    df = df[1:-1]
    return df

def check_format(instance):
    # Check null value
    if np.any(np.isnan(instance)):
        raise Exception("instance contains null data")

def get_instances_and_labels(args, data_subdir, counters):
    if not data_subdir.exists:
        raise Exception("directory not found")

    instances, labels = [], []

    for file in data_subdir.iterdir():
        # Read CSV
        df = pd.read_csv(file)
        df = rename_counter(df)
        df = df[counters]
        df = filter_rows(df)
        df = df.replace(regex="\s", value=0.0)

        # Get label
        iops = df[r"LogicalDisk(D:)\Disk Transfers/sec"].to_numpy().astype(np.float32)
        kernel = np.array([1/args.window_size for _ in range(args.window_size)])
        iops = np.convolve(iops, kernel, mode="valid")
        labels.append(iops)

        # Get instance
        #df = df.drop(r"LogicalDisk(D:)\Disk Transfers/sec", axis=1)
        instance = df.astype("float").to_numpy()
        instance = instance[:-1 * args.window_size + 1]
        check_format(instance)
        instances.append(instance)

    return instances, labels

def preprocess(args, data_dir, counter_path):
    instances, labels = [], []

    counters = get_counter(counter_path)

    data_subdirs = [file for file in data_dir.iterdir() if file.is_dir()]
    for data_subdir in data_subdirs:
        new_instances, new_labels = get_instances_and_labels(args, data_subdir, counters)
        instances.extend(new_instances)
        labels.extend(new_labels)

    # Convert array to numpy array
    data = {
        "instances": np.concatenate(instances, axis=0),
        "labels": np.concatenate(labels, axis=0),
    }

    return data