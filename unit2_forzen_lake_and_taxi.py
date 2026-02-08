import numpy as np
import gymnasium as gym
import random
import imageio
import os
import tqdm
from tqdm import tqdm

grid_name = ['4x4', '8x8'][1]
env_name = ['FrozenLake-v1', 'Taxi-v3'][1]
if env_name == 'FrozenLake-v1':
    env = gym.make(env_name, map_name = grid_name, is_slippery = False, render_mode=['human', 'rgb_array'][1]) # render inside ['human', 'rgb_array', 'ansi', 'rgb_array_list']
else: 
    env = gym.make(env_name, render_mode=['human', 'rgb_array'][1])

state, info = env.reset()

# init the Q-table
state_space = env.observation_space.n  
action_space = env.action_space.n 
def initialize_q_table(state_space, action_space):
    q_table = np.zeros((state_space, action_space)) 
    return q_table
q_table_frozen_lake = initialize_q_table(state_space, action_space)
#print('state_space', state_space, 'action_space', action_space, 'init q-table: ', q_table_frozen_lake)

# policy (RL using off-policy)
# acting: epsilon-greedy ;  updating: greedy
def greedy(q_table, state):
    action = np.argmax(q_table[state][:]) # max_a Q(S_{t_1}, a)
    return action
def epsilon_greedy(q_table, state, epsilon):
    random_number = random.uniform(0, 1) # roll the dice! using the probability to make a choice!
    if random_number > epsilon: # I am sitting at the 1-epsilon area: exploitation
        action = np.argmax(q_table[state][:]) # just the greedy, exploitation
    else:                       # I am sitting at the epsilon area: exploration
        action = env.action_space.sample()
    return action


# hyper-parameters (next step: using optuna)
# how many times of training/testing episode,how many steps in each episode, learning rate, discount rate, epsilon range and probabilitiy
n_training_episodes = 20000
n_eval_episodes = 200
learning_rate = 0.2 # learn from the errors/mistakes
gamma = 0.99 # a bit far sighted
max_epsilon=1
min_epsilon=0.01
if state_space > 16:
    decay_rate = 0.00005 
else:
    decay_rate = 0.01
max_steps = 200 # do not wander forever
eval_seed = []
# training model
def train(n_training_episode, max_epsilon, min_epsilon, decay_rate, max_steps, q_table, env):
    for episode in range(n_training_episodes): 
        # epsilon = np.linspace(max_epsilon, min_epsilon, episode) # linear learning
        epsilon = min_epsilon + (max_epsilon-min_epsilon)*np.exp(-decay_rate*episode)
        state, info = env.reset()
        #print(f'episode: state {state} and info {info}')
        
        step=0
        terminated = False
        truncated = False
        for step in range(max_steps):
            action = epsilon_greedy(q_table, state, epsilon)
            # take action At, get the feedback
            new_state, reward, terminated, truncated, info = env.step(action)
            ##### Begin, 8x8 issue #####
            # hole -1, ice -0.01, target 10
            if env_name =='FrozenLake-v1' and state_space > 16:
                if terminated and reward == 0:
                    reward = -1 
                elif reward == 1:
                    reward = 10
                else:
                    reward = -0.01
            ###### end ######     
            #print(f'step: {step} and action: {action}, new state: {new_state}, reward: {reward}')
            # update the q-table: state-action
            q_table[state][action] = q_table[state][action] + learning_rate*(reward + gamma*np.max(q_table[new_state])-q_table[state][action])

            if terminated or truncated:
                break
            state = new_state
        
    return q_table


q_table_frozen_lake = train(n_training_episodes, max_epsilon, min_epsilon, decay_rate, max_steps, q_table_frozen_lake, env)

print("q_table: \n", q_table_frozen_lake)

# evaluate 
def evaluate_agent(env, max_steps, n_eval_episodes, q, seed):
    episode_rewards = []
    for episode in range(n_eval_episodes):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()
        step = 0
        terminated = False
        truncated = False
        total_rewards_ep = 0
        for steo in range(max_steps):
            action = greedy(q, state)
            new_state, reward, terminated, truncated, into = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward

mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes, q_table_frozen_lake, eval_seed)
print(rf"reward: {mean_reward}+/-{std_reward}")