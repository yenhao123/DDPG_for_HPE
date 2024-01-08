import gym
import gym_pid
import gym_examples
from DDPG import Agent
import numpy as np
import utils

env = gym.make('gym_pid/pid-v0')
#env = gym.make('gym_examples/GridWorld-v0')
agent = Agent(alpha=0.00001, beta=0.0001, input_dims=[3], tau=0.0001, env=env,
              batch_size=64,  layer1_size=256, layer2_size=128, n_actions=3)

#agent.load_models()
# np.random.seed(1)

score_history=[]
for i in range(50):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        print(act)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        print(reward)
        agent.learn()
        score += reward
        obs = new_state
        #env.render()
    raise "1"
    score_history.append(score)

    if i % 10 == 0:
        agent.save_models()
        env.render()

    print('episode ', i, 'score %.2f' % score,
          'trailing 25 games avg %.3f' % np.mean(score_history[-25:]))

