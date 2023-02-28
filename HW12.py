from pyvirtualdisplay import Display
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import torch
import gym
import random
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm

configs = {
    'gamma': 0.5
}

seed = 543  # Do not change this


# virtual_display = Display(visible=0, size=(1400, 900))
# virtual_display.start()


def fix(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.benchmark = False
    torch.backends.deterministic = True


env = gym.make('LunarLander-v2')
initial_state = env.reset()
print(initial_state)
random_action = env.action_space.sample()
print(random_action)
observation, reward, done, info = env.step(random_action)
env.reset()
img = plt.imshow(env.render(mode='rgb_array'))
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)

    img.set_data(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)


# %%
class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)


class PolicyGradientAgent():

    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)

    def forward(self, state):
        return self.network(state)

    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum()  # You don't need to revise this to pass simple baseline (but you can)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob


network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)

agent.network.train()  # Switch network into training mode(network是agent的一个属性，用agent的network属性[同时也是我们之前定义的网络架构]去训练)
EPISODE_PER_BATCH = 5  # update the agent every 5 episode
NUM_BATCH = 500  # totally update the agent for 500 times

avg_total_rewards, avg_final_rewards = [], []

prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:
    log_probs, rewards = [], []
    # reward:单次状态转移的reward
    total_rewards, final_rewards = [], []
    # total_rewards:总的奖励
    # final_rewards:最后一个状态得到的奖励
    # collect trajectory
    for episode in range(EPISODE_PER_BATCH):
        state = env.reset()
        total_reward, total_step = 0, 0
        seq_rewards = []
        while True:
            action, log_prob = agent.sample(state)  # at, log(at|st)
            next_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)  # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            # seq_rewards.append(reward)
            state = next_state
            seq_rewards.append(reward)
            total_reward += reward
            total_step += 1
            # rewards.append(reward)  # change here
            # ! IMPORTANT !
            # Current reward implementation:
            # immediate reward,  given action_list : a1, a2, a3 ......
            # rewards : r1, r2 ,r3 ......

            # medium:change "rewards" to accumulative decaying reward, given action_list : a1,a2,a3, ......
            # rewards:r1+0.99*r2+0.99^2*r3+......, r2+0.99*r3+0.99^2*r4+...... ,  r3+0.99*r4+0.99^2*r5+ ......

            # boss : implement Actor-Critic
            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                for i in range(2, len(seq_rewards) + 1):
                    seq_rewards[-i] += configs['gamma'] * (seq_rewards[-i + 1])
                rewards += seq_rewards
                break

    print(f"rewards looks like ", np.shape(rewards))
    print(f"log_probs looks like", len(log_probs))
    # record training process
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

    # update agent
    # rewards = np.concatenate(rewards, axis=0)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # normalize the reward
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
    print("logs prob looks like", torch.stack(log_probs).size())
    print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())

plt.show()

# %%
plt.plot(avg_total_rewards)
plt.title("Total Rewards")
plt.show()

plt.plot(avg_final_rewards)
plt.title("Final Rewards")
plt.show()

# %%
fix(env, seed)
agent.network.eval()  # set the network into evaluation mode
NUM_OF_TEST = 5  # Do not revise this !!!
test_total_reward = []
action_list = []
for i in range(NUM_OF_TEST):
    actions = []
    state = env.reset()

    img = plt.imshow(env.render(mode='rgb_array'))

    total_reward = 0

    done = False
    while not done:
        action, _ = agent.sample(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)

        total_reward += reward

        img.set_data(env.render(mode='rgb_array'))
        display.display(plt.gcf())
        display.clear_output(wait=True)

    print(total_reward)
    test_total_reward.append(total_reward)

    action_list.append(actions)  # save the result of testing

print(np.mean(test_total_reward))
print("Action list looks like ", action_list)
print("Action list's shape looks like ", np.shape(action_list))

distribution = {}
for actions in action_list:
    for action in actions:
        if action not in distribution.keys():
            distribution[action] = 1
        else:
            distribution[action] += 1
print(distribution)