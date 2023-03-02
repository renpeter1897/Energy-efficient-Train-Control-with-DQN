import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from DQN_test import DQN
from utils import plot_learning_curve, create_directory
from train_env import myTrainEnv
from train_env import mybasic

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=10000)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/train_env/')
parser.add_argument('--images_dir', type=str, default='./output_images/')
parser.add_argument('--info_dir', type=str, default='./stepinfo_csv')
parser.add_argument('--reward_path', type=str, default='./output_images/avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')
parser.add_argument('--energy_path', type=str, default='./output_images/energy.png')

args = parser.parse_args()


def main():
    env = myTrainEnv.TrainEnv()
    agent = DQN(alpha=0.0001, state_dim=4, action_dim=4,
                fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005,
                epsilon=1.0, eps_end=0.05, eps_dec=1e-6, max_size=100000, batch_size=512)
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    create_directory(args.images_dir, sub_dirs=['vs_curve', 'bst_curve'])
    create_directory(args.info_dir, sub_dirs=[])
    total_rewards, avg_rewards, eps_history = [], [], []
    EnergyList = []
    best_reward = -1000
    plt.figure()
    for episode in range(args.max_episodes):
        positions, speed = [0], [0]
        total_reward = 0
        observation = env.reset()
        while True:
            action = agent.choose_acion(observation, isTrain=True)
            if env.state.pos <= 1000:
                action = 0
            observation_, EC, reward, done = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
            if done:
                break
        observation = env.reset()
        while True:
            action = agent.choose_acion(observation)
            if env.state.pos <= 1000:
                action = 0
            observation_, EC, reward, done = env.step(action)
            positions.append(env.state.pos)
            speed.append(env.state.v * 3.6)
            total_reward += reward
            observation = observation_
            if done:
                if total_reward > best_reward:
                    print("best_reward={}, episode{}".format(total_reward, episode+1))
                    mybasic.SL_Grad_curve(env.railway.slpos, env.railway.slval, env.railway.gradpos,
                                          env.railway.gradval)
                    plt.plot(positions, speed)
                    plt.title('{}episodes'.format(episode+1))
                    plt.savefig(args.images_dir + 'bst_curve/bst_curve.png')
                    plt.clf()
                    agent.save_best_models()
                    best_reward = total_reward
                break
        EnergyList.append(EC)
        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
        print('EP:{} reward:{} avg_reward:{} epsilon:{} reason:{} step:{} state:{}'.
              format(episode + 1, total_reward, avg_reward, agent.epsilon, env.reason,
                     env.timestep, env.obs))

        if (episode + 1) % 100 == 0:
            mybasic.SL_Grad_curve(env.railway.slpos, env.railway.slval, env.railway.gradpos, env.railway.gradval)
            plt.plot(positions, speed)
            plt.savefig(args.images_dir + 'vs_curve/{}vs_curve.png'.format(episode+1))
            plt.clf()
            agent.save_models(episode+1)

        if (episode + 1) % 50 == 0:
            df = pd.DataFrame(env.stepinfo)
            df.columns = ['positions', 'speed', 'time', 'action',
                          'Rstop', 'Rtime', 'Renergy', 'Radd', 'Rdis', 'Rspeed']
            df.to_csv(args.info_dir + '/{}step_info.csv'.format(episode+1))

        if (episode + 1) % 500 == 0:
            episodes = [i for i in range(episode+1)]
            plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
            plot_learning_curve(episodes, eps_history, 'Epsilon', 'epsilon', args.epsilon_path)
            plot_learning_curve(episodes, EnergyList, 'Energy', 'energy', args.energy_path)
            plt.clf()

if __name__ == '__main__':
    main()