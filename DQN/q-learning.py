# just reference, not used

import matplotlib.pyplot as plt
import numpy as np

from env import EV_World

#Q-learning


def q_learning(evaluation = False):
    n_agents = 3
    env = EV_World(n_agents)
    # env = MAGridWorld(4, 2)
    # agent = RandomAgent(env)#creating a random agent to explore the given environments 
    obs = env.reset()#resets the environment to its initial configuration
  
    #Intialize parameters
    learning_rate = 0.15 #alpha
    discount_factor = 0.99 #how much weightage to put on future rewards
    det_epsilon = 0.99 # For all states in deterministic environment p(s', r/s, a) = {0, 1}: Either action taken or No action taken


    #Intial state
    current_state = [0, 0] #s1
    action_val = [0,1,2,3]

    #Q table representing 16 rows: one for each state (i.e., 0,1,2,...15) -> (i.e., s1, s2, s3,....s16) and 4 columns: one for each action (i.e., 0,1,2,3) -> (down,up,right,left)
    # (0-15, 0-3) remember the dimension is one less
    q_tables = [np.zeros((16,4)), np.zeros((16,4))] 


    #mapping next_state co-ordinates to q_table co-ordinates
    states = {(0,0): 0, (0,1): 1, (0,2): 2, (0,3): 3,
                    (1,0): 4, (1,1): 5, (1,2): 6, (1,3): 7,
                    (2,0): 8, (2,1): 9, (2,2): 10, (2,3): 11,
                    (3,0): 12, (3,1): 13, (3,2): 14, (3,3): 15} #16 states

    #Empty lists to store values
    
    epsilon_values = []
    
    total_episodes = 10
    epsilon = 1 #multiply by 0.995 for each episode(#after 20 iterations# or terminal state reached)
    decay_factor = (0.01/1)**(1/total_episodes)
    rewards_val = []
    if evaluation == False:
        for episode in range(1, total_episodes+1):
            obs = env.reset()
            current_states = [0, 0]
            total_rewards = [0, 0]
            timestep = 0

            while timestep < 10: #(i.e., considering untill the terminal is reached or 20 timesteps completed)
        
                actions = []
                next_states = []
                for agent in range(2):
                    #e - greedy algorithm
                    rand_num = np.random.random()
                    if epsilon > rand_num:
                        action = np.random.choice(action_val)
                    else:
                        action = np.argmax(q_tables[agent][current_states[agent]]) #action in current state s with max_q value
                    
                    actions.append(action)

                #Taking the action
                next_state_poss, rewards, done, _ = env.step(actions)

                # print("Rewards: ", rewards)
                for pos in next_state_poss:
                    next_states.append(states[tuple(pos)])


            #Choosing action with max Q value
                for agent in range(2):

                    max_q_action = np.argmax(q_tables[agent][next_states[agent]])

                     #Update function
                     # print(actions[agent])
                    q_tables[agent][current_states[agent]][actions[agent]] = q_tables[agent][current_states[agent]][actions[agent]] + learning_rate*(rewards[agent] + discount_factor*q_tables[agent][next_states[agent]][max_q_action] - q_tables[agent][current_states[agent]][actions[agent]])

                    total_rewards[agent] += rewards[agent]

                current_states[0] = next_states[0] #next_state is assigned to current_state
                current_states[1] = next_states[1]
            
                if done[0] and done[1]:
                    done[0] = False
                    done[1] = False
                    break     

            rewards_val.append(total_rewards)
            epsilon_values.append(epsilon) #Append epsilon values in every episode
        

            if epsilon > 0.01: #keeping epsilon in [0.01 - 1] range as if it falls below 0.01 it will exploit more: choosing best actions. We want our agent to explore a bit: choosing random actions
                epsilon = epsilon*decay_factor
            else:
                epsilon = 0.01


            if (episode % 1) == 0:
                print("Episode: {}, epsilon: {}, rewards: {}".format(episode, epsilon, rewards))
                

        #Plotting the results
        #x, y co-ordinates
        x = [episode for episode in range(1, total_episodes+1)]
        ye = epsilon_values
        yr1 = [rewards_val[episode][0] for episode in range(total_episodes)]
        yr2 = [rewards_val[episode][1] for episode in range(total_episodes)]
        # yr2 = total_rewards[1]

        #Plots showing episodes vs epsilon, episodes vs rewards
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,4))
        #episodes vs epsilon
        ax1.plot(x, ye)
        ax1.set_title("Epsilon decay")

        #episodes vs rewards
        ax2.plot(x,yr1)
        ax2.set_title("Agent1: Rewards per episode")

        #episodes vs rewards
        ax3.plot(x,yr2)
        ax3.set_title("Agent2: Rewards per episode")

    else:
        eval_rewards_val = []

        for i in range(10):
            obs = env.reset()
            current_states = [0, 0]
            total_rewards = [0, 0]
            timestep = 0
            agent1_states = [13]
            agent2_states = [16]

            while timestep < 10: #(i.e., considering untill the terminal is reached or 20 timesteps completed)
            
                actions = []
                next_states = []
                
                for agent in range(2):
                    action = np.argmax(q_tables[agent][current_states[agent]]) #action in current state s with max_q value
                    actions.append(action)

                #Taking the action
                next_state_poss, rewards, done, _ = env.step(actions)

                # print("Rewards: ", rewards)
                for pos in next_state_poss:
                    next_states.append(states[tuple(pos)])

                #Choosing action with max Q value
                for agent in range(2):
                    max_q_action = np.argmax(q_tables[agent][next_states[agent]])

                    #Update function
                    q_tables[agent][current_states[agent]][actions[agent]] = q_tables[agent][current_states[agent]][actions[agent]] + learning_rate*(rewards[agent] + discount_factor*q_tables[agent][next_states[agent]][max_q_action] - q_tables[agent][current_states[agent]][actions[agent]])

                    total_rewards[agent] += rewards[agent]
                
                agent1_states.append(next_states[0] + 1)
                agent2_states.append(next_states[1] + 1)
                current_states[0] = next_states[0] #next_state is assigned to current_state
                current_states[1] = next_states[1]
                
                if done[0] and done[1]:
                    done[0] = False
                    done[1] = False
                    break     

            eval_rewards_val.append(total_rewards)
            env.render()
            
            # print("Evaluation rewards: ", eval_rewards_val)
            print("Agent 1 route: ", agent1_states)
            print("Agent 2 route: ", agent2_states) 
