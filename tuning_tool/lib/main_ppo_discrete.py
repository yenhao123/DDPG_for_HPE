import sys
# change home_path
HOME_PATH = r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool"
sys.path.append(HOME_PATH)
import gym
import gym_pid
from pathlib import Path
import pickle
import numpy as np
import math
import utils
import subprocess
import os
import json
import random

from model.PPO_action_masking import PPOAgent

LOAD_MODEL = False
LOG_DIR = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\log")
CONFIG_PATH = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\config\config.json")
POWERSHELL_PATH = r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\main.ps1"
N_EPISODES = 50
N_STATES = 33
N_ACTIONS = 8
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

def output_action(x, idx2action):
    idx = np.argmax(x)
    return idx2action[idx], idx

def get_action_mapping():
    qd_list = [2, 4, 8, 16, 32]
    mnpd_list = list(range(0, 31, 5))
    idx2action = []
    for qd in qd_list:
        for mnpd in mnpd_list:
            for mc in [0, 1]:
                for pc in [0, 1]:
                    for dpo in [0, 1]:
                        for irp in [0, 1]:
                            for dwc in [0, 1]:
                                for smartpath_ac in range(7):
                                    idx2action.append([qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac])
    return idx2action

if __name__ == "__main__":
    env = gym.make('gym_pid/pid-v0')

    # Initialize agent
    ### environment hyperparameters ###
    has_continuous_action_space=False
    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    save_model_freq = 1

    ### model hyperparameters ###
    max_ep_len = 10
    update_timestep = max_ep_len
    state_dim=N_STATES
    # Action Mapping {idx: (qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac)}
    idx2action = get_action_mapping()
    N_ACTIONS = len(idx2action)
    print("Action space: {}".format(N_ACTIONS))
    action_dim=N_ACTIONS
    lr_actor=3e-4
    lr_critic=1e-3 
    gamma=0.99 # discount factor
    k_epochs=80 # update policy for K epochs
    eps_clip=0.2

    ppo_agent = PPOAgent(state_dim, action_dim, lr_actor, lr_critic, 
                        gamma, k_epochs, eps_clip, has_continuous_action_space, action_std)


    if LOAD_MODEL:
        checkpoint_path = r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\param\ppo\PPO.pth"
        ppo_agent.load(checkpoint_path)

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
    time_step = 0

    for i in range(N_EPISODES):
        obs = env.reset()
        log_obs(obs, "default", "default", state_path)
        done = False
        info_per_eposide = []
        while not done:
            is_random = np.random.choice(["random", "nonrandom"], p=[EXPLORATION_RATE, 1-EXPLORATION_RATE])
            if is_random == "random":
                mc, pc, dpo, irp, dwc, qd, mnpd, smartpath_ac = randomize()
                act = [qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac]
            else:
                act, act_prob = ppo_agent.select_action(obs)
                action = idx2action[act]
                qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac = action
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
            
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            time_step += 1
            
            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()
            
            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
            obs = new_state
            env.render()

            # Log data
            log_obs(new_state, reward, is_random, state_path)
            log_action(act_prob, LOG_DIR / "action.txt")
            info_per_eposide.append((action,new_state[0]))

        action_per_eposide = np.array([action for action, _ in info_per_eposide])
        score_per_eposide = np.array([score for _, score in info_per_eposide])
        max_score = np.max(score_per_eposide)
        recommended_action = action_per_eposide[np.argmax(score_per_eposide)]
        print(f"Episode {i} finished, max reward: {max_score}, recommended action: {recommended_action}")

        if (i+1) % save_model_freq == 0:
            checkpoint_dir = LOG_DIR / "checkpoint"
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoint_dir / "PPO_{}.pth".format(i)
            ppo_agent.save(checkpoint_path)
        env.render()
        
    env.close()