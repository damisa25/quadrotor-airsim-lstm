import gym
from gym import spaces

from airsim.AirSimClient import *
from airgym.envs.airsim_env import AirSimEnv
from airgym.envs.drone_env import AirSimDroneEnv

from stable_baselines import PPO2
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.bench.monitor import Monitor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.callbacks import EvalCallback, BaseCallback, EveryNTimesteps, CheckpointCallback

import matplotlib.pyplot as plt
from tensorboard import program
import numpy as np
import math
import time
import os


# connect to the AirSim simulator
client = MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=1,
                destination=np.array([13.650886, 100.492384, 22.0]),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
# env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = PPO2(
    MlpLstmPolicy,
    env,
    learning_rate=0.00025,
    verbose=1,
    max_grad_norm=10,
    tensorboard_log='./tb_logs/',
)

checkpoint_callback = CheckpointCallback(save_freq=500, save_path='./checkpoint/ppo2_25000_steps/',
                                         name_prefix='lstm_policy')

time_steps = 25000
model.learn(
    total_timesteps=int(time_steps),
    log_interval=5,
    tb_log_name="ppo2_25000_steps",
    callback=checkpoint_callback,
)

# Save policy weights
# model.save("model/dqn_airsim_drone_policy")
model.save("model/ppo2_25000_steps")


# time_steps = 100
# model = DQN(
#     "CnnPolicy",
#     env,
#     learning_rate=0.00025,
#     verbose=1,
#     batch_size=32,
#     train_freq=4,
#     target_update_interval=10000,
#     learning_starts=10000,
#     buffer_size=50000,
#     max_grad_norm=10,
#     exploration_fraction=0.1,
#     exploration_final_eps=0.01,
#     tensorboard_log='./tb_logs/',
# )

# checkpoint_callback = CheckpointCallback(save_freq=100, save_path='./test_checkpoint/dqn_4actions_10000_steps/',
#                                          name_prefix='dqn_policy')

# model.learn(
#     total_timesteps=int(time_steps),
#     log_interval=5,
#     tb_log_name="test" + str(time_steps) + "_steps",
#     callback=checkpoint_callback,
# )