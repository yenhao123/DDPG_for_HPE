import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np

class PidEnv(gym.Env):
    def __init__(self, sample_rate=1, setpoint=50):
        self.observation_space = spaces.Box(0, 1, shape=(3,))
        self.action_space = spaces.Dict({
            "proportional":spaces.Box(0, 1),
            "integral":spaces.Box(0, 1),
            "derivative":spaces.Box(0, 1),
            })
        self.sample_rate = sample_rate
        self.setpoint = setpoint
        self.error = self.setpoint
        self.proportional = 0
        self.integral = 0
        self.derivative = 0
        self.last_error = self.error
        self.currpoint = [0, 0]
        self.kp = 0.5
        self.ki = 0.5
        self.kd = 0.5
        self.n = 200 # Simulation points


    def step(self, action):
        self.currpoint = [0, 0]
        self.xhistory = [0]
        self.yhistory = [0]
        # spilit action
        self.kp = action[0] # Increasing p term reduces rise time
        self.ki = action[1]
        self.kd = action[2] # Increasing d term improves stability and decreases overshoot
        done = False

        # get observation & reward
        while(self.currpoint[0]< self.n and done == False ):
            # max x axis of n points 
            self.proportional = self.kp * self.error
            self.integral += self.ki * self.error * self.sample_rate
            self.derivative = self.kd * (self.error - self.last_error) / self.sample_rate

            curr_input = self.proportional + self.integral + self.derivative

            self.last_error = self.error
            self.currpoint[1] += curr_input
            self.currpoint[0] += 1 
            self.error = self.setpoint - float(self.currpoint[1])
            self.xhistory.append(self.currpoint[0])
            self.yhistory.append(self.currpoint[1])
            if(self.currpoint[1] == self.yhistory[self.currpoint[0]-1]):
                done = True
            print(self.currpoint)
            print(self.yhistory[self.currpoint[0]-1])
    

        self.state = [self.kp, self.ki, self.kd]
        reward = -abs(self.error) - 0.005 * self.currpoint[0]
        if reward > -10:
            reward += 10
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return self.state
    
    def reset(self):
        self.error = self.setpoint
        self.proportional = 0
        self.integral = 0
        self.derivative = 0
        self.last_error = self.error
        self.currpoint = [0,0]
        self.kp = 0.5
        self.ki = 0.5
        self.kd = 0.5
        self.continous = False
        return self.step(np.array([0,0,0]))[0]

    def render(self):
        print("Error: "+str(self.error))
        print("Proportional Term: "+str(self.proportional))
        print("Integral Term: "+str(self.integral))
        print("Derivative Term: "+str(self.derivative))
        plt.plot(self.xhistory, self.yhistory)
        plt.show()
