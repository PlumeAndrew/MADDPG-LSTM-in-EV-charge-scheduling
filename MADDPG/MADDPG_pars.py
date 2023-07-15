import argparse

import torch as T

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser("MADDPG experiments for EV charging scheduling")

    # core training parameters
    parser.add_argument("--device", default=device, help="training device ")
    parser.add_argument("--episodes", type=int, default=5000, help="training episodes")
    parser.add_argument("--agents", type=int, default=3, help="num of agents")
    parser.add_argument("--max_episode_steps", type=int, default=24, help="max steps for each episode")
    parser.add_argument("--batch_size", type=int, default=256, help="number of episodes from buffer")
    parser.add_argument("--memory_size", type=int, default=1000000, help="number of data stored in the memory")
    # parser.add_argument("--log", type=str, default="./log/epoch_{}/{}", help="tensorboard log dir")
    parser.add_argument("--price_file", type=str, default="data/2019-04.csv", help="ele price file dir")
    parser.add_argument("--price_source", type=str, default="lstm", help="ele price used for training, select in: random, real, lstm")

    # networks settings
    parser.add_argument("--fc_1", type=int, default=64, help="dims of the first full connected layers")
    parser.add_argument("--fc_2", type=int, default=64, help="dims of the second full connected layers")
    parser.add_argument("--lr_a", type=float, default=0.001, help="learning rate for adam optimizer in Actor")
    parser.add_argument("--lr_c", type=float, default=0.001, help="learning rate for adam optimizer in Critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for agent to foresee the state")
    parser.add_argument("--tau", type=int, default=0.01, help="randomness of buffer sampling in soft update")

    # env settings
    parser.add_argument("--max_battery_capacity", type=float, default=1.5, help="max of the EV battery capacity")
    parser.add_argument("--max_charge_power", type=float, default=0.1)
    parser.add_argument("--min_charge_power", type=float, default=-0.1)
    parser.add_argument("--base_load", type=float, default=1.0, help="basic power system load")
    parser.add_argument("--load_restriction", type=float, default=0.20, help="load reward limited by the persentage of charging load in base load")
    parser.add_argument("--price_factor", type=float, default=10, help="discount of price reward")
    parser.add_argument("--load_factor", type=float, default=3, help="discount of load reward")
    parser.add_argument("--soc_factor", type=float, default=0.5, help="discount of soc reward")

    return parser.parse_args()
