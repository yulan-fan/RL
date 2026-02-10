import torch
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStack
from unit3_helper import QNetwork, ReplayBuffer, select_action, optimize_model # Import NN helpers
import numpy as np

# env setup
env = gym.make("ALE/SpaceInvaders-v5", frameskip=1, render_mode='rgb_array') # with NO internal skipping
env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4)
env = FrameStack(env, num_stack=4)

# load the policy from helper function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Hmm, Meine CPU
policy_net = QNetwork(env.action_space.n).to(device) # learning...
target_net = QNetwork(env.action_space.n).to(device) # frozen snapshot for stability (like a notebook)
target_net.load_state_dict(policy_net.state_dict())  # sync
optimizer = optim.Adam(policy_net.parameters(), lr=1e-4) # optimize the weights

# for each episode, load the memory
n_episodes = 100
nsteps = 10000
batch_size = 32
gamma = 0.99
target_update = 1000
step_done = 0
eps_end = 0.05
eps_start = 1.0
eps_decay = 100000 
opt_step = 4

memory = ReplayBuffer(100000)
for episode in range(n_episodes):
    
    state, _ = env.reset()
    # no memory will be discard through episode, so move it outside the loop
    # memory = ReplayBuffer(10000, state, action, reward, next_state, done)
    
    for t in range(nsteps):
        # which action   
        epsilon = max(eps_end, eps_start - (step_done / eps_decay))
        action = select_action(policy_net, state, epsilon, device, env.action_space)
        # action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # memorize
        memory.push(state, action, next_state, reward, done)
        # update
        state = next_state
        step_done += 1
        # train
        if len(memory) > batch_size and step_done % opt_step == 0: # optimize every 4 steps
            optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma, device)

        if step_done % target_update == 0: 
            target_net.load_state_dict(policy_net.state_dict()) # update / buffer to target net
            print(f"Target Network Updated at step {step_done} with epsilon {epsilon}")
        if done: 
            break
