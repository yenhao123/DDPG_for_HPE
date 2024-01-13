import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

PERF_LOG_PATH = r'C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\fio\performance_log.csv'

class PidEnv(gym.Env):
    def __init__(self, max_steps=100):
        super(PidEnv, self).__init__()
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
        # state: score, action
        self.initial_state = np.array([self.throughput, 0, 0 ,0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.observation = None
        self.done = False
        self.ideal = np.array([1200.0], dtype=np.float32) #ideal values for action space 

    def reset(self):
        self.done = False
        self.counter = 0
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
        print(self.counter)
        # Compute the new state based on the action (random formulae for now)
        # TODO: Replace with actual reward function
        df = pd.read_csv(PERF_LOG_PATH)
        df.replace(regex="\s", value=0.0, inplace=True)
        df = df.astype('float32')
        df = df.loc[df["LogicalDisk(D:)\Disk Transfers/sec"]!=0]
        self.throughput = df["LogicalDisk(D:)\Disk Transfers/sec"].mean()
        self.bandwidth = df["LogicalDisk(D:)\Disk Bytes/sec"].mean()

        print("Throughput: ", self.throughput)
        print("Bandwidth: ", self.bandwidth)

        self.observation = np.array([self.throughput, qd, mnpd, mc, pc, dpo, irp, dwc, smartpath_ac], dtype=np.float32)
        objective = np.array([self.throughput], dtype=np.float32)
        reward = objective - self.ideal

        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        return self.observation

    def render(self, mode='human'):
        print (f'Throughput: {self.throughput}, Bandwidth: {self.bandwidth}')