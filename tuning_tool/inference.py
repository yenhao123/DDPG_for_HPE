import gym
import gym_pid
from DDPG import Agent
from pathlib import Path
import pickle
import numpy as np
import utils
import subprocess
import os
import json

LOAD_MODEL = True
N_INFERENCE_ITERATIONS = 5
LOG_DIR = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\log")
CONFIG_PATH = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\config\config.json")
POWERSHELL_PATH = r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\main.ps1"


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

def new_file(path):
    # 检查文件是否存在
    if os.path.exists(path):
        # 如果存在，删除文件
        os.remove(path)
        print(f"File '{path}' deleted.")
    else:
        print(f"File '{path}' does not exist.") 

def randomize():
    import random
    mc = random.randint(0, 1)
    pc = random.randint(0, 1)
    dpo = random.randint(0, 1)
    irp = random.randint(0, 1)
    dwc = random.randint(0, 1)
    qd = random.randint(2, 32)
    mnpd = random.randint(0, 60)
    smartpath_ac = random.randint(0, 7)
    return mc, pc, dpo, irp, dwc, qd, mnpd, smartpath_ac

# just used in original range between -1 and 1, e.g., tanh
def scale_action(x, low, high):
    x = (x + 1) / (1 + 1) * (high - low) + low
    return x

if __name__ == "__main__":
    env = gym.make('gym_pid/pid-v0')
    agent = Agent(alpha=0.00001, beta=0.0001, input_dims=[9], tau=0.0001, env=env,
                batch_size=64,  layer1_size=256, layer2_size=128, n_actions=8)

    if LOAD_MODEL:
        agent.load_models()

    
    score_history = []
    infer_dir = LOG_DIR / "inference"
    infer_dir.mkdir(parents=True, exist_ok=True)
    log_path = infer_dir / "state.txt"
    new_file(log_path)

    obs = env.reset()
    done = False
    score = 0
    for i in range(N_INFERENCE_ITERATIONS):
        act = agent.choose_action(obs)
        
        # Preprocess action
        qd_list = [2, 4, 8, 16, 32]
        qd_idx = int(scale_action(act[0], 0, 5))
        qd = qd_list[qd_idx]
        mnpd = int(scale_action(act[1], 0, 60))
        mc, pc, dpo, irp, dwc, smartpath_ac = is_bigger_than_zero(act[2:8])
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

        # State order: [qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac]
        new_state, reward, done, info = env.step(action)
        agent.remember(obs, act, reward, new_state, int(done))
        obs = new_state
        with log_path.open("a") as f:
            f.write(str(new_state).replace("\n", "") + "\n")
                
        score_history.append(score)
        env.render()