# ref: https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from DQN_pars import parse_args
from env import EV_World

args = parse_args()

# Hyper Parameters
BATCH_SIZE = args.batch_size
LR = args.lr                   # learning rate
EPSILON = args.epsilon              # greedy policy
GAMMA = args.gamma                 # reward discount
TARGET_REPLACE_ITER = args.target_replace_iter   # target update frequency
MEMORY_CAPACITY = args.memory_size
N_EPISODES = args.episodes
MAX_TIME_STEPS = args.max_episode_steps
env = EV_World(1)
power_lsit = [-0.08,-0.04,0.04,0.08] # discrete action refers to power

N_ACTIONS = env.action_space.n 
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

writer = SummaryWriter("log/DQN/{}/{}".format(
    N_EPISODES, time.strftime("%m%d_%H%M")))

class Net(nn.Module):


    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):


    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.loss = 0

    def choose_action(self, x):
        # modify for EV_World
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)

        action = power_lsit[action]
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.loss = loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()
print("\nModel Init...")

# main train loop
for i_episode in range(N_EPISODES):
    s = env.reset()
    total_reward = 0
    total_steps = 0
    for t in range(MAX_TIME_STEPS):
        # Take the random actions in the beginning for the better exploration
        # if total_steps < 5000:  
        #     a = env.action_space.sample()
        # else:
        #     a = dqn.choose_action(s)
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)

        # modify the reward, not necessary
        # x, x_dot, theta, theta_dot = s_
        # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        # r = r1 + r2

        dqn.store_transition(s, a, r, s_)

        # exp replay
        # if dqn.memory_counter > MEMORY_CAPACITY:
        if total_steps % 100 == 0:
            dqn.learn()

        # state trans
        s = s_

        # record reward
        total_reward += r

        total_steps += 1

    if i_episode % 100 == 0 and i_episode > 0:
        print("Episode: {}, Total Rewards: {}".format(
                i_episode, total_reward))

    writer.add_scalar(tag="episode_rewards",
                scalar_value=total_reward ,
                global_step=i_episode)
    writer.add_scalar(tag="DQN_loss",
                scalar_value=dqn.loss,
                global_step=i_episode)
    writer.add_scalar(tag="eval/ele_price",
                          scalar_value=env.ele_price,
                          global_step=i_episode)
    writer.add_scalar(tag="eval/action",
                          scalar_value=a, 
                          global_step=i_episode)

