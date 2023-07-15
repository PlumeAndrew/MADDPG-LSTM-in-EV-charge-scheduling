import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces

# from lstm import predict
from DQN_pars import parse_args

args = parse_args()

# EV agent = [soc, soc_ideal, start_time, end_time]
# base_load = 1.0
# max_battery_capacity = 1.5
# min_charge_power = -10
# max_charge_power = 10
# load_restriction = 0.3
# load_factor = 0.7
# soc_factor = 0.5


# ver 1.4 for DQN
class EV_World(gym.Env):
    # one agent 

    def __init__(self, n_agents, max_time_steps=24):
        self.n_agents = n_agents
        self.date = 0
        self._time_step = 0  # time pointer
        self._max_time_steps = max_time_steps

        # load ele price file
        self.file = np.loadtxt(args.price_file,
                               delimiter=",",
                               encoding="UTF-8")
        # self.lstm_model = torch.load("model/lstm/lstm.pt")
        # self.predict_price = predict(self.lstm_model).flatten()
       
        self.ele_price = self._update_ele_price_real()

        self.action_space = spaces.Discrete(4) # action means charging power, range [P_min, P_max], but discrete
        # self.action_space = spaces.Box(
        #     low=args.min_charge_power, high=args.max_charge_power,
        #     shape=(1, ))  # action means charging power, range [P_min, P_max]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.n_agents,1))  # soc

    def step(self, action):
        obs_n = []
        reward = 0
        done_n = []
        info_n = {"n": []}

        # reward_n = [0 for _ in range(self.n_agents)]

        for i in range(self.n_agents):
            # charge EV
            if (self._time_step >= self.start_time_list[i]
                    and self._time_step <= self.end_time_list[i]):
                self._update_agent_state(i, action)
                reward = self._get_reward(i, action)

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
        # self.ele_price = self._update_ele_price_random()
        self.ele_price = self._update_ele_price_real()
        # self.ele_price = self._update_ele_price_LSTM()
        # print(self.ele_price)

        self._time_step += 1

        return obs_n, reward, done_n, info_n

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

    def _get_reward(self, agent, action):
        # price reward
        price_reward = -args.price_factor * self.ele_price * action
        # punish reward
        if action / args.base_load > args.load_restriction:
            punish_reward = -args.load_factor * self.ele_price * action
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
        
