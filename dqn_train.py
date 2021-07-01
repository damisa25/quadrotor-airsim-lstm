import gym
from gym import spaces

from airsim.AirSimClient import *
from airgym.envs.airsim_env import AirSimEnv
from airgym.envs.drone_env import AirSimDroneEnv

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, EveryNTimesteps, CheckpointCallback

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
                image_shape=(84, 84, 1),
                destination=np.array([300,0,-30]),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
# env = VecTransposeImage(env)

# Initialize RL algorithm type and parameters
model = DQN(
    "MultiInputPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=10000,
    buffer_size=50000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    tensorboard_log='./tb_logs/',
)

checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./checkpoint/v11_dqn_multiInput_4actions_2obs_simpleRF_100000_steps/',
                                         name_prefix='dqn_policy')

time_steps = 100000
model.learn(
    total_timesteps=int(time_steps),
    log_interval=5,
    tb_log_name="v11_dqn_multiInput_4actions_2obs_simpleRF_100000_steps",
    callback=checkpoint_callback,
)

# Save policy weights
# model.save("model/dqn_airsim_drone_policy")
model.save("model/v11_dqn_multiInput_4actions_2obs_simpleRF_100000_steps")


# time_steps = 100000
# model = DQN(
#     "MultiInputPolicy",
#     env,
#     learning_rate=0.00025,
#     verbose=1,
#     batch_size=32,
#     train_freq=4,
#     target_update_interval=200,
#     learning_starts=200,
#     buffer_size=10000,
#     max_grad_norm=10,
#     exploration_fraction=0.1,
#     exploration_final_eps=0.01,
#     tensorboard_log='./test_tb_logs/',
# )

# checkpoint_callback = CheckpointCallback(save_freq=100, save_path='./test_checkpoint/dqn_4actions_10000_steps/',
#                                          name_prefix='dqn_policy')

# model.learn(
#     total_timesteps=int(time_steps),
#     log_interval=5,
#     tb_log_name="test" + str(time_steps) + "_steps",
#     callback=checkpoint_callback,
# )