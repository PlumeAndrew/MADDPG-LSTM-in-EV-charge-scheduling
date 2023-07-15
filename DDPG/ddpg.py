# ref: https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/6.DDPG/DDPG.py

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from DDPG_pars import parse_args
from env import EV_World

args = parse_args()


class Actor(nn.Module):
    def __init__(self, state_dim, fc1_dims, fc2_dims, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, fc1_dims)
        self.l2 = nn.Linear(fc1_dims, fc2_dims)
        self.l3 = nn.Linear(fc2_dims, action_dim)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, s):
        s = F.relu(self.l1(s))
        s = F.relu(self.l2(s))
        a = self.max_action * torch.tanh(self.l3(s))  # [-max,max]
        return a


class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    def __init__(self, state_dim, action_dim, fc1_dims, fc2_dims):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, fc1_dims)
        self.l2 = nn.Linear(fc1_dims, fc2_dims)
        self.l3 = nn.Linear(fc2_dims, 1)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, s, a):
        q = F.relu(self.l1(torch.cat([s, a], 1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class ReplayBuffer(object):
    def __init__(self, max_size, state_dim, action_dim, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.count = 0
        # self.size = 0

        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        # self.size = min(self.size + 1, self.max_size)  # Record the number of transitions

    def sample(self):
        index = np.random.choice(self.max_size, size=self.batch_size, replace=False)  # Randomly sampling
        # batch_s = torch.tensor(self.s[index], dtype=torch.float)
        # batch_a = torch.tensor(self.a[index], dtype=torch.float)
        # batch_r = torch.tensor(self.r[index], dtype=torch.float)
        # batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        # batch_dw = torch.tensor(self.dw[index], dtype=torch.float)
        batch_s = self.s[index]
        batch_a = self.a[index]
        batch_r = self.r[index]
        batch_s_ = self.s_[index]
        batch_dw = self.dw[index]

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        # hypper parameters
        self.fc_1 = args.fc_1  # The number of neurons in hidden layers of the neural network
        self.fc_2 = args.fc_2
        self.batch_size = args.batch_size  # batch size
        self.GAMMA = args.gamma  # discount factor
        self.TAU = args.tau  # Softly update the target network
        self.device = args.device
        # self.hidden_width = 256 
        # self.batch_size = 256  
        # self.GAMMA = 0.99  
        # self.TAU = 0.005  
        # self.lr = 3e-4  

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(state_dim, self.fc_1, self.fc_2, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim, self.fc_1, self.fc_2)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_a)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr_c)

        self.actor_loss = 0
        self.critic_loss = 0

        self.MseLoss = nn.MSELoss()

    def choose_action(self, state, noise_en=True):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = self.actor.forward(state)
        # action = self.actor(state).data.numpy().flatten()

        # Add Gaussian noise to actions for exploration
        noise = torch.rand(self.action_dim).to(self.device)
        if noise_en:
            action = action + noise
        else:
            action = action
        
        # adjust the action into an appropriate value
        # soc range[0,1] and we hope action range[-0.1,0.1]
        action = action / 10

        return action.detach().cpu().numpy()[0]

    def learn(self, relay_buffer):
        batch_s, batch_a, batch_r, batch_s_, batch_dw = relay_buffer.sample()  # Sample a batch

        batch_s = torch.tensor(batch_s, dtype=torch.float).to(self.device)
        batch_a = torch.tensor(batch_a, dtype=torch.float).to(self.device)
        batch_r = torch.tensor(batch_r, dtype=torch.float).to(self.device)
        batch_s_ = torch.tensor(batch_s_, dtype=torch.float).to(self.device)
        batch_dw = torch.tensor(batch_dw, dtype=torch.float).to(self.device)

        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_))
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a)
        critic_loss = self.MseLoss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = -self.critic(batch_s, self.actor(batch_s)).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        self.actor_loss = actor_loss
        self.critic_loss = critic_loss


def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s)  # We do not add noise when evaluating
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


def reward_adapter(r, env_index):
    if env_index == 0:  # Pendulum-v1
        r = (r + 8) / 8
    elif env_index == 1:  # BipedalWalker-v3
        if r <= -100:
            r = -1
    return r


if __name__ == '__main__':
    max_episode_steps = args.max_episode_steps # Maximum number of steps per episode
    n_agents = args.agents
    n_episodes = args.episodes # Maximum number of training steps
    batch_size = args.batch_size
    max_memory_capacity = args.memory_size

    env = EV_World(n_agents)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_action = float(env.action_space.high[0])
    noise_std = 0.1 * max_action  # the std of Gaussian noise for exploration
      
    # print("env={}".format(env_name[env_index]))
    # print("state_dim={}".format(state_dim))
    # print("action_dim={}".format(action_dim))
    # print("max_action={}".format(max_action))
    # print("max_episode_steps={}".format(max_episode_steps))

    agent = DDPG(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(max_memory_capacity, state_dim, action_dim, batch_size)
    # Build a tensorboard
    # writer = SummaryWriter(log_dir='runs/DDPG/DDPG_env_{}_number_{}_seed_{}'.format(env_name[env_index], number, seed))
    writer = SummaryWriter("log/DDPG/{}/{}".format(
        n_episodes, time.strftime("%m%d_%H%M")))

    # random_steps = 1000  # Take the random actions in the beginning for the better exploration
    # update_freq = 50  # Take 50 steps,then update the networks 50 times
    # evaluate_freq = 1e3  # Evaluate the policy every 'evaluate_freq' steps
    # evaluate_num = 0  # Record the number of evaluations
    # evaluate_rewards = []  # Record the rewards during the evaluating

    total_steps = 0  # Record the total steps during the training
    # while total_steps < max_train_steps:
    for i_episode in range(n_episodes):
        s = env.reset()
        total_reward = 0
        time_steps = 0
        # done = False
        # while not done:
        for t in range(max_episode_steps):
            # episode_steps += 1
            if total_steps < 10000:  # Take the random actions in the beginning for the better exploration
                a = env.action_space.sample()
            else:
                a = agent.choose_action(s)

            s_, r, done, _ = env.step(a)

            # Adjust rewards for better performance
            # r = reward_adapter(r, env_index)
               
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and time_steps != max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(s, a, r, s_, dw)  # Store the transition

            # exp replay
            if total_steps % 100 == 0:
                agent.learn(replay_buffer)

            # state trans
            s = s_

            # Take 50 steps,then update the networks 50 times
            # if total_steps >= random_steps and total_steps % update_freq == 0:
            #     for _ in range(update_freq):
            #         agent.learn(replay_buffer)

            # record reward
            total_reward += r[0]
            #TODO to compare with 3 agents 

            # Evaluate the policy every 'evaluate_freq' steps
            # if (total_steps + 1) % evaluate_freq == 0:
            #     evaluate_num += 1
            #     evaluate_reward = evaluate_policy(env_evaluate, agent)
            #     evaluate_rewards.append(evaluate_reward)
            #     print("evaluate_num:{} \t evaluate_reward:{}".format(evaluate_num, evaluate_reward))
            #     writer.add_scalar('step_rewards_{}'.format(env_name[env_index]), evaluate_reward, global_step=total_steps)
            #     # Save the rewards
            #     if evaluate_num % 10 == 0:
            #         np.save('./data_train/DDPG_env_{}_number_{}_seed_{}.npy'.format(env_name[env_index], number, seed), np.array(evaluate_rewards))
            time_steps += 1
            total_steps += 1

        if i_episode % 100 == 0 and i_episode > 0:
            print("Episode: {}, Total Rewards: {}".format(
                    i_episode, total_reward))

        writer.add_scalar(tag="episode_rewards",
                    scalar_value=total_reward ,
                    global_step=i_episode)
        writer.add_scalar(tag="actor_loss",
                            scalar_value=agent.actor_loss,
                            global_step=i_episode)
        writer.add_scalar(tag="critic_loss",
                            scalar_value=agent.critic_loss,
                            global_step=i_episode)
        writer.add_scalar(tag="eval/ele_price",
                          scalar_value=env.ele_price,
                          global_step=i_episode)
        writer.add_scalar(tag="eval/action",
                          scalar_value=a, 
                          global_step=i_episode)
        
