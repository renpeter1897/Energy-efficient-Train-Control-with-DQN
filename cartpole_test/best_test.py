import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from DQN_test import DQN
from utils import plot_learning_curve, create_directory
from train_env import myTrainEnv
from train_env import mybasic

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/train_env/')
args = parser.parse_args()

env = myTrainEnv.TrainEnv()
agent = DQN(alpha=0.0001, state_dim=4, action_dim=4,
            fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005,
            epsilon=1.0, eps_end=0.05, eps_dec=1e-6, max_size=1000000, batch_size=512)

for i in range(2):
    observation = env.reset()
    if i == 0:
        agent.load_best_models()
    else:
        agent.load_models(6000)
    positions, speed = [0], [0]
    total_reward = 0
    while True:
        action = agent.choose_acion(observation, isTrain=False)
        if env.state.pos <= 1000:
            action = 0
        observation_, EC, reward, done = env.step(action)
        positions.append(env.state.pos)
        speed.append(env.state.v * 3.6)
        total_reward += reward
        observation = observation_
        if done:
            mybasic.SL_Grad_curve(env.railway.slpos, env.railway.slval, env.railway.gradpos,
                                  env.railway.gradval)
            plt.plot(positions, speed)
            plt.show()
            break
