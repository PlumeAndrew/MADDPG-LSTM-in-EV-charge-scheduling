import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces

import lstm
from MADDPG_pars import parse_args

args = parse_args()

# EV agent = [soc, soc_ideal, start_time, end_time]
# base_load = 1.0
# max_battery_capacity = 1.5
# min_charge_power = -10
# max_charge_power = 10
# load_restriction = 0.3
# load_factor = 0.7
# soc_factor = 0.5


# ver 1.3 for MADDPG
class EV_World(gym.Env):
    # metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, n_agents, max_time_steps=24):
        self.n_agents = n_agents
        self.date = 0
        self._time_step = 0  # time pointer
        self._max_time_steps = max_time_steps
        self.price_source = args.price_source
        # load ele price file
        self.file = np.loadtxt(args.price_file,
                               delimiter=',',
                               encoding='UTF-8')
        self.lstm_model = torch.load("model/lstm/lstm_2000.pt")
        self.predict_price = lstm.predict(self.lstm_model).flatten()

        self.ele_price = self._update_ele_price_real()

        self.action_space = spaces.Box(
            low=args.min_charge_power, high=args.max_charge_power,
            shape=(1, ))  # action means charging power, range [P_min, P_max]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(1 + self.n_agents,
                   1))  # one-dim vector? [soc_1,soc_2, ... ,ele_price]

    def step(self, actions):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {"n": []}

        reward_n = [0 for _ in range(self.n_agents)]

        for i in range(self.n_agents):
            # charge EV when EV is online
            if (self._time_step >= self.start_time_list[i]
                    and self._time_step <= self.end_time_list[i]):
                self._update_agent_state(i, actions[i])
                reward_n[i] = self._get_reward(i, actions)
            # reward_n[i] = self._get_reward(i, actions)

            # if EV leaves, task done
            if self._time_step > self.end_time_list[i]:
                self.done[i] = True
            else:
                self.done[i] = False

        # obs = [soc_1, soc_2,..., soc_n_agents, ele_price]
        # obs[n_agents] = ele_price
        obs_n = self.soc_list
        done_n = self.done

        # update ele price for next time step
        if self.price_source is "random":
            self.ele_price = self._update_ele_price_random()
        elif self.price_source is "real":
            self.ele_price = self._update_ele_price_real()
        elif self.price_source is "lstm":
            self.ele_price = self._update_ele_price_LSTM()
        # print(self.ele_price)

        self._time_step += 1

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        self._time_step = 0
        self.done = [False for i in range(self.n_agents)]

        self.soc_list = []
        self.soc_ideal_list = []
        self.start_time_list = []
        self.end_time_list = []

        # init each agent
        for i in range(self.n_agents):
            start_time = np.random.randint(23, high=None, size=None,
                                           dtype="l")  # init start_time
            end_time = np.random.randint(start_time,
                                         high=23,
                                         size=None,
                                         dtype="l")  # init end_time
            soc_ideal = np.random.normal(loc=0.5, scale=0.16,
                                         size=None)  # init ideal soc ideal
            soc = np.random.normal(loc=0.5, scale=0.16, size=None)  # init soc

            # record all agent states
            self.soc_list.append(soc)
            self.soc_ideal_list.append(soc_ideal)
            self.start_time_list.append(start_time)
            self.end_time_list.append(end_time)

        # obs = [soc_1, soc_2,..., soc_n_agents, ele_price]
        observation = self.soc_list.copy()
        obs = np.array(observation)
        # obs = np.append(obs, self.ele_price)
        # obs = np.concatenate((observation, price), axis=1)

        return obs

    def _update_agent_state(self, agent, action):
        # this func is used to update EV soc
        self.soc_list[agent] += action  # charge EV with current Power=action

    def _get_reward(self, agent, actions):
        # price reward
        price_reward = -args.price_factor * self.ele_price * actions[agent]
        # punish reward
        if sum(actions) / args.base_load > args.load_restriction:
            punish_reward = -args.load_factor * self.ele_price * actions[agent]
        else:
            punish_reward = 0

        # soc reward
        if self._time_step == self.end_time_list[agent]:
            soc_reward = -args.soc_factor * pow(
                args.max_battery_capacity *
                (self.soc_ideal_list[agent] - self.soc_list[agent]), 2)
        else:
            soc_reward = 0

        # all reward for one agent
        reward = price_reward + punish_reward + soc_reward

        return reward

    def _update_ele_price_random(self, low=0.1, high=0.3):
        # produce random ele_price between [0.1,0.3]
        a = high - low
        b = high - a
        next_price = np.random.rand() * a + b
        return next_price

    def _update_ele_price_real(self):
        # load real ele price 
        if self._time_step % 24 == 0:
            self.date += 1
            if self.date % 30 == 0:
                self.date = 0

        return self.file[self.date * self._max_time_steps + self._time_step]

    def _update_ele_price_LSTM(self):
        # use LSTM net to predict ele_price
        if self._time_step % 24 == 0:
            self.date += 1
            if self.date % 30 == 0:
                self.date = 0
        price = self.date * self._max_time_steps + self._time_step
        if price >=714 :
            price -= 6

        # notice that predict_price is only 714 len because of the sliding window
        return self.predict_price[price]
        
    # def render(self):
    #     state = np.copy(self.world)
    #     for i in range(self.n_agents):
    #         state[tuple(self.agent_pos[i])] = 1 - (0.1 * i)
    #         state[tuple(self.goal_pos[i])] = 1 - (0.1 * i)
    #     return plt.imshow(state)

    # ver 1.1
    # def step(self, time_step, action_n):
    #     states_n = []
    #     reward_n = []

    #     for i, agent in enumerate(self.agent_list):
    #         # charge EV with current Power
    #         if time_step >= agent.start_time and time_step <= agent.end_time:
    #             agent.soc += action_n[i]

    #     for agent in self.agent_list:
    #         # record observation and reward for each agent
    #         states_n.append()
    #         reward_n.append(self.get_reward(agent, action_n[agent]))

    #     # compute total reward
    #     total_reward = np.sum(reward_n)
    #     reward_n = [total_reward] * self.n_agents

    #     return states_n, reward_n

    # def reset(self, agent_list):
    #     # init each agent
    #     for i in agent_list:
    #         start_time = np.random.randint(23, high=None, size=None, dtype="l")
    #         end_time = np.random.randint(start_time, high=23, size=None, dtype="l")
    #         soc_ideal = np.random.normal(loc=0.5, scale=1.0, size=None)
    #         agent_list[i].soc = 0.5  # init soc
    #         agent_list[i].soc_ideal = soc_ideal  # init ideal soc
    #         agent_list[i].start_time = start_time  # init start_time
    #         agent_list[i].end_time = end_time  # init end_time

    # def get_reward(self, agent, action):
    #     # price reward
    #     price_reward = -ele_price * action
    #     # punish reward
    #     if sum(ele_price) / base_load > load_restriction:
    #         punish_reward = -load_factor * ele_price * action
    #     else:
    #         punish_reward = 0

    #     # TODO soc reward
    #     # if end_time == max_episode_step:
    #     #     soc_reward = soc_factor * (max_battery_capacity * (soc_ideal - soc)) ^ 2
    #     # else:
    #     #     soc_reward = 0
    #     soc_reward = (
    #         soc_factor * (max_battery_capacity * (agent.soc_ideal - agent.soc)) ^ 2
    #     )

    #     # total reward for one agent
    #     reward = price_reward + punish_reward + soc_reward

    #     return reward
