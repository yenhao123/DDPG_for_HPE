from gym.envs.registration import register

register(
    id='gym_pid/pid-v0',
    entry_point='gym_pid.envs:PidEnv',
)