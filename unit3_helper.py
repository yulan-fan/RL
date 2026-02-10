import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple

# Here is my 'rational brain'
class QNetwork(nn.Module):

    def __init__(self, action_size):
        super(QNetwork, self).__init__()

        # Input: (4, 84, 84) ---- 4 stacked grayscale frames
        self.conv = nn.Sequential(
            # converlutional layer (input-channels, output-channels, kernals, shrinkage), activation
            # output  = ( (inputs - kernal_size ) / stride ) + 1
            nn.Conv2d(4, 32,  kernel_size=8, stride=4), nn.ReLU(), # ( 84 - 8 ) / 4 + 1 = 20    
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), # ( 20 - 4 ) / 2 + 1 = 9
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU() # ( 9  - 3 ) / 1 + 1 = 7
        ) # return the (channel, width, height) -> (64, 7, 7) 

        # fully connected layer
        # 64 x 7 x 7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, action_size)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1) # flatten 64 x 7 x 7
        return self.fc(x) 


# Memory, here is my "emotional brain"
class ReplayBuffer:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)  # ATTENTION: RANDOM! Even though some memories might contribute a tiny to current state
        states, actions,next_states, rewards, dones = zip(*batch) 
        return (np.stack(states), 
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.stack(next_states),
                np.array(dones, np.uint8) # 0: continue  1: game over
               )
    def __len__(self):
        return len(self.memory)

# The epsilon-greedy policy
# A modification of 'epsilon_greedy()' in unit2_forzen_lake_and_taxi.py
# ATTENTION: one state -> one decision
def select_action(policy_net, state, epsilon, device, action_space):

    # prepare the state
    # NN works with float -> change the pixels from 0-255 to 0.0-1.0
    state_v = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device) / 255.0

    if random.random() > epsilon: 
        # disables gradient calculation to run faster, thanks gemini.
        with torch.no_grad():
            return policy_net(state_v).argmax().item() # I am sitting at the 1-epsilon area: exploitation
        
    else: # I am sitting at the epsilon area: exploration
        # actually, to get the action size, 
        return action_space.sample() 


# Deinfe the structure of one single memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

def optimize_model(memory, policy_net, target_net, optimizer, batch_size, gamma, device):
    
    if len(memory) < batch_size:
        return # Not enough memories to learn yet!
    
    # 1. sample a 'random' transition from the memory
    # transitions = memory.sample(batch_size)  # ATTENTION: RANDOM! Even though some memories might contribute a tiny to current state
    #batch = Transition(*zip(*transitions)) # do not have to unzip it again
    b_state, b_action, b_reward, b_next_state, b_done = memory.sample(batch_size)
    
    # 2. convert raw data into PyTorch Tensors
    state_batch  = torch.FloatTensor(np.array(b_state)).to(device) / 255.0 
    action_batch = torch.LongTensor(b_action).unsqueeze(1).to(device) # to a column
    reward_batch = torch.FloatTensor(b_reward).to(device)
    next_state_batch = torch.FloatTensor(np.array(b_next_state)).to(device) / 255.0 
    done_batch   = torch.FloatTensor(b_done).to(device) # targrt = reward + (future * (1-done))

    # 3. Loss 
    # 3.1 current q-value (prediction)
    current_q_value = policy_net(state_batch).gather(1, action_batch) # 1: column (action)

    # 3.2 next "best possible" Target value  
    with torch.no_grad():
        max_next_q_value = target_net(next_state_batch).max(1)[0] # max(1): best posiible [0]: value
    expected_q_value = reward_batch + (gamma * max_next_q_value * (1-done_batch)) # " TD target "

    loss = F.mse_loss(current_q_value.squeeze(), expected_q_value)

    # Finally, comes to the "OPTIMIZATION"
    # based on the loss, trace back and update the weight
    optimizer.zero_grad() # keep current 32 memories 
    loss.backward() # back-ptopagation to get the "Gradient" (local minimum)
    # Hmm, First order Taylor approximation !!!  -> Flat world assumption !!! 
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    optimizer.step() # update 
