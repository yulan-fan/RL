import os
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

import torch


# Create the env
# env = gym.make("LunarLander-v2", render_mode='human') # A single Standard environment 
env = make_vec_env("LunarLander-v2", n_envs=16) # Multiple / Parallel 

# model
model = PPO(
    "MlpPolicy", # mlp policy of the model
    env,
    #learning_rate = 0.5, 
    n_steps = 512,  # session (n_step * n_envs)
    batch_size = 64, 
    n_epochs = 4, 
    gamma = 0.999,
    gae_lambda = 0.98,  
    #clip_range = 0.5, 
    #clip_range_vf = None, 
    #normalize_advantage = False,
    ent_coef = 0.01, 
    verbose = 1, 
    )

# Train the model
model.learn(total_timesteps=300000) # learning hours (There might be a plateau ?!)
# Save the model
model_name = "ppo-LunarLander-v2"
model.save(model_name)

# Evaluate the agent
eval_env = Monitor(gym.make("LunarLander-v2", render_mode='human')) # human, allow to show the moving
mean_reward, std_reward = evaluate_policy(
    model, 
    eval_env, 
    n_eval_episodes=30, 
    deterministic=True,
    render = True, # corresponding to human
    #return_episode_rewards = True,
    )
#print(f"episode rewards: {episode_rewards}")
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

# show the lander moving
obs, info = eval_env.reset()

for i in range(30):

    # get the prediction
    action, _states = model.predict(obs, deterministic=True)
    ## How the agent think ? (The policy)
    # 1. get the action probability
    obs_tensor = torch.tensor(obs).unsqueeze(0).to("cpu")
    distribution = model.policy.get_distribution(obs_tensor)
    probs = distribution.distribution.probs.detach().numpy()[0]
    # 2. get the value of the prection (agent's think it's safe -> high value)
    value = model.policy.predict_values(obs_tensor).item()

    action_names = ["Nothing", "Fire Left", "Fire Main", "Fire Right"]
    obs, reward, done, truncated, info = eval_env.step(action)
    #print(f"Step {i}: obs (X={obs[0]:.2f}, Y={obs[1]:.2f}) -> Action {action_names[action]}")
    print(f"Step {i}, Prob: Nothing {probs[0]:.2f}, Left {probs[1]:.2f}, Main {probs[2]:.2f}, Right:{probs[3]:.2f} -> Action {action_names[action]}, reward: {reward:.3f}")
    if done or truncated: 
        break

eval_env.close()


'''
print("_____OBSERVATION SPACE_____ \n") # (x, y, vx, vy, theta, omega, Leg_L, Leg_R)
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # get an random observation

print("_____ACTION SPACE_______\n") # (0: nothing, 1: left, 2: main, 3: right)
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # take an random action
'''
# Sum of the reward of each timestep (>=200)
# for each timestep, the reward is
#   closer/further based on (x,y);      (obs space)
#   slower/faster based on (vx, vy);    (obs space)  
#   tilt (angle);                       (obs space) 
#   +10 by Leg_L or Leg_R;              (obs sapce)
#   +0.03 by firing L or R;             (action space) 
#   +0.3 by firing main;                (action space)
#   -100/+100 crashing / landing



