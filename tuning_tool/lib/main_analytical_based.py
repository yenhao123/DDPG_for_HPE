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
from concurrent import futures
import grpc
import portpicker

PACKAGE_PATH = r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\packages\vizier"
sys.path.append(PACKAGE_PATH)
from vizier.service import clients
from vizier.service import pyvizier as vz
from vizier.service import vizier_server
from vizier.service import vizier_service_pb2_grpc
from vizier._src.algorithms.designers.random import RandomDesigner
from vizier._src.algorithms.designers.emukit import EmukitDesigner


LOAD_MODEL = False
LOG_DIR = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\log")
PARAM_DIR = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\param")
CONFIG_PATH = Path(r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\config\config.json")
POWERSHELL_PATH = r"C:\Users\Administrator\Desktop\Master_Thesis\tuning_tool\env_communicate\main.ps1"
NUM_STEPS = 100
N_ACTIONS = 8
EXPLORATION_RATE = 0
SEED = 110

def log_obs(new_state, reward, is_random, log_path):
    log_data = np.append(new_state, reward)
    log_data = np.append(log_data, is_random)
    log_data = str(log_data).replace("\n", "")
    with log_path.open("a") as f:
        f.write(log_data + "\n")

def log_action(action, log_path):
    with log_path.open("a") as f:
        f.write(str(action) + "\n")

def new_file(path):
    # 检查文件是否存在
    if os.path.exists(path):
        # 如果存在，删除文件
        os.remove(path)
        print(f"File '{path}' deleted.")

def main():
    """Trains the custom environment using random actions for a given number of steps and episodes
    """

    env = gym.make('gym_pid/pid-v0')
    fitness_hist = {}
    problem = vz.ProblemStatement()
    
    qd_list = [2 ** i for i in range(1, 6)]
    problem.search_space.select_root().add_discrete_param(name='QD', feasible_values=qd_list)
    mnpd_list = list(range(0, 31, 5))
    problem.search_space.select_root().add_discrete_param(name='MNPD', feasible_values=mnpd_list)
    problem.search_space.select_root().add_discrete_param(name='MemoryCompression', feasible_values=[0, 1])
    problem.search_space.select_root().add_discrete_param(name='PageCombining', feasible_values=[0, 1])
    problem.search_space.select_root().add_discrete_param(name='DPO', feasible_values=[0, 1])
    problem.search_space.select_root().add_discrete_param(name='IRP', feasible_values=[0, 1])
    problem.search_space.select_root().add_discrete_param(name='DWC', feasible_values=[0, 1])
    problem.search_space.select_root().add_categorical_param(name='Smartpath_ac', feasible_values=[0, 1, 2, 3, 4, 5, 6])

    problem.metric_information.append(
        vz.MetricInformation(
            name='Reward', goal=vz.ObjectiveMetricGoal.MAXIMIZE))




    study_config = vz.StudyConfig.from_problem(problem)
    random_designer = RandomDesigner(problem.search_space, seed = SEED)
    bo_designer = EmukitDesigner(problem)

    port = portpicker.pick_unused_port()
    address = f'localhost:{port}'

    # Setup server.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))

    # Setup Vizier Service.
    servicer = vizier_server.VizierService()
    vizier_service_pb2_grpc.add_VizierServiceServicer_to_server(servicer, server)
    server.add_secure_port(address, grpc.local_server_credentials())

    # Start the server.
    server.start()

    clients.environment_variables.service_endpoint = address  # Server address.
    study = clients.Study.from_study_config(
        study_config, owner='owner', study_id='example_study_id')

    '''
    # experiment name
    exp_name = "_num_steps_" + str(num_steps) + "_num_episodes_" + str(num_episodes)

    # append logs to base path
    log_path = os.path.join(summary_dir, 'random_search_logs', reward_formulation, exp_name)
    

    # get the current working directory and append the exp name
    global traject_dir
    traject_dir = os.path.join(summary_dir, traject_dir, reward_formulation, exp_name)
        


    # check if log_path exists else create it
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    if use_envlogger:
        if not os.path.exists(traject_dir):
            os.makedirs(traject_dir)
    env = wrap_in_envlogger(env, traject_dir)
    '''
    # Logging 
    state_path, action_path = LOG_DIR / "state.txt", LOG_DIR / "action.txt"
    new_file(state_path)
    new_file(action_path)
    
    # """
    # This loop runs for num_steps * num_episodes iterations.
    # """
    env.reset()

    count = 0
    # Designer TODO
    suggestions = bo_designer.suggest(count=NUM_STEPS)

    for suggestion in suggestions:
        count += 1
        # Get agent
        qd = str(suggestion.parameters['QD'])
        mnpd = str(suggestion.parameters['MNPD'])
        mc = str(suggestion.parameters["MemoryCompression"])
        pc = str(suggestion.parameters["PageCombining"])
        dpo = str(suggestion.parameters["DPO"])
        irp = str(suggestion.parameters["IRP"])
        dwc = str(suggestion.parameters["DWC"])
        smartpath_ac = str(suggestion.parameters['Smartpath_ac'])
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
        # Iterate with environment
        new_state, reward, done, info = env.step(action)
        
        # Save trajectory and reward to log_path
        fitness_hist['reward'] = reward
        fitness_hist['action'] = action
        fitness_hist['obs'] = new_state
        if count == NUM_STEPS:
            done = True

        # Log data
        log_obs(new_state, reward, "False", state_path)
        log_action(action, LOG_DIR / "action.txt")
        
        # Update agent
        final_measurement = vz.Measurement({'Reward': reward})
        suggestion = suggestion.to_trial()
        suggestion.complete(final_measurement)

if __name__ == '__main__':
  main()