import sys
# change home_path
HOME_PATH = r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool"
sys.path.append(HOME_PATH)
import gym
import gym_pid
from model.DDPG import DDPGAgent
from pathlib import Path
import pickle
import numpy as np
import math
import utils
import subprocess
import os
import json
import random


LOG_DIR = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\log")
CONFIG_PATH = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\config\config.json")
POWERSHELL_PATH = r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\main.ps1"
N_ITERATIONS = 500
N_STATES = 33

# np.random.seed(1)

def is_bigger_than_zero(x):
    for i in range(len(x)):
        if x[i] > 0:
            x[i] = 1
        else:
            x[i] = 0
    return x

def tune_config_windows(config):
    with CONFIG_PATH.open('w') as file:
        json.dump(config, file)

    result = subprocess.run(["powershell", POWERSHELL_PATH], capture_output=True, text=True)

    if result.returncode == 0:
        print("Successfully execute PowerShell")
        print(result.stdout)
    else:
        print("Failed to execute PowerShell")
        print(result.stderr)
        raise "1"

def new_dir(path):
    # 检查文件夹是否存在
    if not os.path.exists(path):
        # 如果不存在，创建文件夹
        os.makedirs(path)
        print(f"Directory '{path}' created.")

def clean_file(path):
    # 检查文件是否存在
    if os.path.exists(path):
        # 如果存在，删除文件
        os.remove(path)
        print(f"File '{path}' deleted.")

def randomize():
    mc = random.randint(0, 1)
    pc = random.randint(0, 1)
    dpo = random.randint(0, 1)
    irp = random.randint(0, 1)
    dwc = random.randint(0, 1)
    qd_list = [2, 4, 8, 16, 32]
    qd_idx = random.randint(0, len(qd_list)-1)
    qd = qd_list[qd_idx]
    mnpd_list = list(range(0, 31, 5))
    mnpd_idx = random.randint(0, len(mnpd_list)-1)
    mnpd = mnpd_list[mnpd_idx]
    smartpath_ac = random.randint(0, 6)
    return mc, pc, dpo, irp, dwc, qd, mnpd, smartpath_ac

# just used in original range between -1 and 1, e.g., tanh
def scale_action(x, low, high):
    x = round((x + 1) / (1 + 1) * (high - low) + low)
    if x > high:
        x = high
    elif x < low:
        x = low
    return x

def log_obs(new_state, reward, is_random, log_path):
    log_data = np.append(new_state, reward)
    log_data = np.append(log_data, is_random)
    log_data = str(log_data).replace("\n", "")
    with log_path.open("a") as f:
        f.write(log_data + "\n")

def log_action(action, log_path):
    with log_path.open("a") as f:
        f.write(str(action) + "\n")

if __name__ == "__main__":
    env = gym.make('gym_pid/pid-v0')

    # Log Setting
    new_dir(LOG_DIR)
    fio_log_dir = LOG_DIR / "fio"
    new_dir(fio_log_dir)
    logman_log_dir = LOG_DIR / "logman"
    new_dir(logman_log_dir)
    state_path, action_path = LOG_DIR / "state.txt", LOG_DIR / "action.txt"
    clean_file(state_path)
    clean_file(action_path)

    # Iteratively Configuration Tuning
    for i in range(N_ITERATIONS):
        obs = env.reset()
        log_obs(obs, "start", "start", state_path)
        done = False
        env.render()