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
import time


LOAD_MODEL = False
LOG_DIR = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\log")
CONFIG_PATH = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\config\config.json")
POWERSHELL_PATH = r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\main.ps1"
N_EPISODES = 200
N_STATES = 33
EXPLORATION_RATE = 0

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

    # Hyperparameters Setting
    param_dir = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\param\ddpg")
    agent = DDPGAgent(alpha=1e-5, beta=1e-4, input_dims=[N_STATES], tau=1e-4, env=env,
                batch_size=8,  layer1_size=256, layer2_size=128, n_actions=8, chkpt_dir=param_dir)
    
    if LOAD_MODEL:
        agent.load_models()

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
    for i in range(N_EPISODES):
        obs = env.reset()
        log_obs(obs, "start", "start", state_path)
        done = False
        while not done:
            is_random = np.random.choice(["random", "nonrandom"], p=[EXPLORATION_RATE, 1-EXPLORATION_RATE])
            if is_random == "random":
                mc, pc, dpo, irp, dwc, qd, mnpd, smartpath_ac = randomize()
                act = [qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac]
            else:
                act = agent.choose_action(obs)
                # Preprocess action
                ## discrete option
                qd_list = [2, 4, 8, 16, 32]
                qd_idx = scale_action(act[0], 0, len(qd_list)-1)
                qd = qd_list[qd_idx]
                mnpd_list = list(range(0, 31, 5))
                mnpd_idx = scale_action(act[1], 0, len(mnpd_list)-1)
                mnpd = mnpd_list[mnpd_idx]
                ## categorical option
                smartpath_ac = scale_action(act[7], 0, 6)
                ## binary option
                mc, pc, dpo, irp, dwc = is_bigger_than_zero(act[2:7])


            action = {
                "qd" : float(qd),
                "mnpd" : float(mnpd),
                "mc" : float(mc),
                "pc" : float(pc),
                "dpo" : float(dpo),
                "irp" : float(irp),
                "dwc" : float(dwc),
                "smartpath_ac" : float(smartpath_ac)
            }
            print("Suggested Parameters")
            print(f"queue depth:{qd}, mnpd:{mnpd}, memory compression:{mc}, page combining:{pc}, dpo:{dpo}, irp:{irp}, dwc:{dwc}, smartpath_ac:{smartpath_ac}")
            config = {
            "configuration" : [int(mc), int(pc), int(dpo), int(irp), int(dwc), int(qd), int(mnpd), int(smartpath_ac)]
            }
            tune_config_windows(config)

            # state order: [throughput, qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac]
            # action order: [qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac]
            new_state, reward, done, info = env.step(action)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            obs = new_state
            env.render()
            
            # Log data
            log_obs(new_state, reward, is_random, state_path)
            log_action(act, LOG_DIR / "action.txt")

        agent.save_models()
        env.render()
        if i % 5 == 0:
            time.sleep(1800)