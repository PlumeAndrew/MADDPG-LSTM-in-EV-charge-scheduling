import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from buffer import MultiAgentReplayBuffer
from env import EV_World
from maddpg import MADDPG
from MADDPG_pars import parse_args

args = parse_args()


def obs_list_to_state_vector(observation):
    state = observation
    state = np.array([])
    for obs in observation:
        state = np.append(state, obs)
    return state


if __name__ == "__main__":
    max_episode_steps = args.max_episode_steps
    n_agents = args.agents
    n_episodes = args.episodes
    batch_size = args.batch_size
    max_memory_capacity = args.memory_size
    exploration = True

    env = EV_World(n_agents, max_episode_steps)
    obs = env.reset()
    writer = SummaryWriter("./log/MADDPG/{}_{}/{}".format(
        args.price_source, n_episodes, time.strftime("%m%d_%H%M")))

    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(1)
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 1

    maddpg_agents = MADDPG(
        actor_dims,
        critic_dims,
        n_agents,
        n_actions,
        chkpt_dir="model/maddpg/",
    )

    memory = MultiAgentReplayBuffer(max_memory_capacity, critic_dims,
                                    actor_dims, n_actions, n_agents,
                                    batch_size)

    # main loop
    total_steps = 0
    reward_history = []
    reward_best = 0

    # each episode
    for i_episode in range(n_episodes):
        obs = env.reset()
        a_loss = []
        c_loss = []
        obs_next = []
        done = [False] * n_agents
        time_steps = 0  # time pointer
        total_reward = 0

        # each step
        for t in range(max_episode_steps):
            # each agent selects own action under current policy
            if exploration and total_steps < 20000:
                actions = []
                for agent in range(n_agents):
                    actions.append(env.action_space.sample())
                actions = np.array(actions)
            else:
                actions = maddpg_agents.choose_action(obs)

            # excute aciton and get reward and next states
            obs_, rewards, done, info_n = env.step(actions)

            # record state
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if time_steps >= max_episode_steps:
                done = [True] * n_agents

            # store <s, a, r, s_> in buffer
            memory.store_transition(obs, state, actions, rewards, obs_, state_,
                                    done)

            # experience replay and then update policy every 100 steps
            if total_steps % 100 == 0:
                maddpg_agents.learn(memory)

            # state trans
            obs = obs_

            # compute total reward
            total_reward += sum(rewards)

            time_steps += 1
            total_steps += 1
        # end for

        reward_history.append(total_reward)
        reward_average = np.mean(reward_history[-100:])
        maddpg_agents.episodes_done += 1
        # if i_episode % (n_episodes // 100) == 0 and i_episode > 0:
        if i_episode % 100 == 0 and i_episode > 0:
            print("Episode: {}, Total Rewards: {}".format(
                i_episode, total_reward))

        # if i_episode % 1000 == 0 and i_episode > 0:
        #     maddpg_agents.save_checkpoint()
        # another update method is to use average reward to measure the model
        # if reward_average > reward_best:
        # maddpg_agents.save_checkpoint()
        # reward_best = reward_average

        writer.add_scalar(tag="episode_rewards",
                          scalar_value=total_reward,
                          global_step=i_episode)
        writer.add_scalar(tag="actor_loss",
                          scalar_value=maddpg_agents.actor_loss,
                          global_step=i_episode)
        writer.add_scalar(tag="critic_loss",
                          scalar_value=maddpg_agents.critic_loss,
                          global_step=i_episode)
        writer.add_scalar(tag="eval/ele_price",
                          scalar_value=env.ele_price,
                          global_step=i_episode)
        writer.add_scalar(tag="eval/action",
                          scalar_value=actions[0], 
                          global_step=i_episode)
        # writer.close()
    # end for
