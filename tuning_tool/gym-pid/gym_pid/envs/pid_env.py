import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import os
from datetime import datetime
import math

import json

PERF_LOG_PATH = r'C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\log\logman\performance_log.csv'
PERF_FIO_LOG_PATH = r'C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\log\fio\state_fio.json'
THROUGHPUT_IDX = 0
CONFIG_PATH = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\config\config.json")
POWERSHELL_PATH = r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\main.ps1"

class PidEnv(gym.Env):
    def __init__(self, max_steps=10):
        super(PidEnv, self).__init__()
        # TODO: Define action and observation space
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(9,))
        
        self.action_space = spaces.Dict({
          "qd": spaces.Discrete(5),
          "mnpd": spaces.Box(low = 0, high = 60, dtype = int),
          "mc": spaces.Discrete(2),
          "pc": spaces.Discrete(2),
          "dpo": spaces.Discrete(2),
          "irp": spaces.Discrete(2),
          "dwc": spaces.Discrete(2),
          "smartpath_ac": spaces.Discrete(8),
        })
         
        self.max_steps = max_steps

        self.counter = 0
        self.bandwidth = 0
        self.throughput = 0
        self.initial_state = 0
        self.observation = None
        self.done = False
        self.ideal = np.array([100.0], dtype=np.float32) #ideal values for action space 
        self.prev_obj = np.array([0.0], dtype=np.float32)
        self.default_obj = np.array([0.0], dtype=np.float32)
        self.action_history = []

    def reset(self):
        self.done = False
        self.counter = 1
        self.action_history = []
        self.initial_state = self._get_init_state()
        return self.initial_state
        
    # Input : action; Output : observation & reward &
    def step(self, action):
        qd = action['qd']
        mnpd = action['mnpd']
        mc = action['mc']
        pc = action['pc'] 
        dpo = action['dpo']
        irp = action['irp']
        dwc = action['dwc']
        smartpath_ac = action['smartpath_ac'] 

        action = np.array([qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac])
        if (self.counter == self.max_steps):
            self.done = True
            print("Maximum steps reached")
        else:
            self.counter += 1

        # get the reward
        fio_metrics, logman_metrics = self.observe()
        option = np.array([qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac], dtype=np.float32)
        self.observation = np.concatenate([fio_metrics, logman_metrics, option], axis=None)
        objective = np.array([fio_metrics[0]], dtype=np.float32)
        reward = self._get_reward(objective)
        
        self.prev_obj = objective
        self.action_history.append(action)
        print("Reward: ", reward)
        return self._get_obs(), reward, self.done, {}

    def _calucate_reward(self, delta0, deltat):
        if delta0 > 0:
            _reward = ((1+delta0)**2-1) * math.fabs(1+deltat)
        else:
            _reward = - ((1-delta0)**2-1) * math.fabs(1-deltat)

        if _reward > 0 and deltat < 0:
            _reward = 0
        return _reward

    def _get_reward(self, objective):
        delta_0_tp = float((objective[0] - self.default_obj[0]))/self.default_obj[0]
        delta_t_tp = float((objective[0] - self.prev_obj[0]))/self.prev_obj[0]
        
        tp_reward = self._calucate_reward(delta_0_tp, delta_t_tp)
        return tp_reward

    def observe(self):
        # Get system states from fio tool
        with open(PERF_FIO_LOG_PATH, 'r') as f:
            json_data = json.load(f)
        self.throughput = json_data["jobs"][0]["read"]["iops"] + json_data["jobs"][0]["write"]["iops"]
        self.bandwidth = json_data["jobs"][0]["read"]["bw"] + json_data["jobs"][0]["write"]["bw"]
        print("Throughput: ", self.throughput)
        print("Bandwidth(kb): ", self.bandwidth)
        rename_file(PERF_FIO_LOG_PATH)
        fio_metrics = [self.throughput, self.bandwidth]

        # Get system states from logman tool
        df = pd.read_csv(PERF_LOG_PATH)
        df.replace(regex="\s", value=0.0, inplace=True)
        df = df.astype('float32')
        df = df.loc[df["LogicalDisk(D:)\Disk Transfers/sec"]!=0]

        ld_cols = [col for col in df.columns if "LogicalDisk" in col]
        df = df[ld_cols]
        # throughput 移到第一行
        first_column = df.pop("LogicalDisk(D:)\Disk Transfers/sec")
        df.insert(0, "LogicalDisk(D:)\Disk Transfers/sec", first_column)
        logman_metrics = df.mean().tolist()

        return fio_metrics, logman_metrics

    def _get_obs(self):
        return self.observation

    def render(self, mode='human'):
        print (f'Throughput: {self.throughput}, Bandwidth: {self.bandwidth}')

    def _get_init_state(self):
        #file_server_act = [-1, 15, 0, 0, 1, 1, 1, 2]
        act = [-1, 0, 0, 0, 0, 0, 0, 0]
        qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac = act
        config = {
            "configuration" : [int(mc), int(pc), int(dpo), int(irp), int(dwc), int(qd), int(mnpd), int(smartpath_ac)]
        }

        self.tune_config_windows(config)

        fio_metrics, logman_metrics  = self.observe()
        self.default_obj = np.array([fio_metrics[0]], dtype=np.float32)
        self.prev_obj = np.array([fio_metrics[0]], dtype=np.float32)
        initial_state = np.concatenate([fio_metrics, logman_metrics , np.array(act)], axis=None)
        return initial_state

    def get_init_state(self):
        return self.initial_state

    def tune_config_windows(self, config):
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

    # Input : action; Output : observation & reward &
    def sythetetic_step(self, action):
        qd = action['qd']
        mnpd = action['mnpd']
        mc = action['mc']
        pc = action['pc'] 
        dpo = action['dpo']
        irp = action['irp']
        dwc = action['dwc']
        smartpath_ac = action['smartpath_ac'] 

        action = np.array([qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac])
        if (self.counter == self.max_steps):
            self.done = True
            print("Maximum steps reached")
        else:
            self.counter += 1

        # get the reward
        fio_metrics, logman_metrics = self.synthetic_observe()
        option = np.array([qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac], dtype=np.float32)
        self.observation = np.concatenate([fio_metrics, logman_metrics, option], axis=None)
        objective = np.array([fio_metrics[0]], dtype=np.float32)
        reward = self._get_reward(objective)
        
        self.prev_obj = objective
        self.action_history.append(action)
        print("Reward: ", reward)
        return self._get_obs(), reward, self.done, {}

    def synthetic_observe(self):
        perf_sythetic_dir = Path(r'C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\fio\synthetic')
        
        # Get system states from fio tool
        perf_synthetic_fio_dir = perf_sythetic_dir / "fio"
        files = list(perf_synthetic_fio_dir.iterdir())
        file_idx = round(np.random.uniform(low=0, high=len(files)-1, size=1)[0])
        with open(files[file_idx], 'r') as f:
            json_data = json.load(f)
        self.throughput = json_data["jobs"][0]["read"]["iops"] + json_data["jobs"][0]["write"]["iops"]
        self.bandwidth = json_data["jobs"][0]["read"]["bw"] + json_data["jobs"][0]["write"]["bw"]
        print("Throughput: ", self.throughput)
        print("Bandwidth(kb): ", self.bandwidth)
        fio_metrics = [self.throughput, self.bandwidth]

        # Get system states from logman tool
        perf_synthetic_logman_dir = perf_sythetic_dir / "logman"
        files = list(Path(perf_synthetic_logman_dir).iterdir())
        file_idx = round(np.random.uniform(low=0, high=len(files)-1, size=1)[0])
        df = pd.read_csv(files[file_idx])
        df.replace(regex="\s", value=0.0, inplace=True)
        df = df.astype('float32')
        df = df.loc[df["LogicalDisk(D:)\Disk Transfers/sec"]!=0]

        ld_cols = [col for col in df.columns if "LogicalDisk" in col]
        df = df[ld_cols]
        first_column = df.pop("LogicalDisk(D:)\Disk Transfers/sec")
        df.insert(0, "LogicalDisk(D:)\Disk Transfers/sec", first_column)
        logman_metrics = df.mean().tolist()

        return fio_metrics, logman_metrics



def rename_file(old_name):
    try:
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Extract file extension from old_name (if any)
        base_name, file_extension = os.path.splitext(old_name)

        # Create a new name with timestamp
        new_name = f"{base_name}_{timestamp}{file_extension}"

        # Rename the file
        os.rename(old_name, new_name)

        print(f"File '{old_name}' has been renamed to '{new_name}'.")
    except FileNotFoundError:
        print(f"Error: File '{old_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")