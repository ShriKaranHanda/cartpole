import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import math

# Define a transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        # Simplified - removed priorities for speed
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        # Simple random sampling - much faster
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, 
                 buffer_size=100000,
                 batch_size=128,
                 gamma=0.99,
                 lr=0.0003,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.997,  # Slower decay
                 tau=0.001,
                 update_every=4,
                 lr_decay=0.9999):  # Learning rate decay factor
        
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.initial_lr = lr  # Store initial learning rate
        self.lr_decay = lr_decay  # Learning rate decay factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.update_every = update_every
        
        # Q-Network
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set target network to evaluation mode
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=1e-5)  # Added weight decay
        # Create scheduler for learning rate decay
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.lr_decay)
        
        self.memory = ReplayBuffer(buffer_size)
        
        # Initialize step counter
        self.t_step = 0
        self.episode_step = 0
        
    def step(self, state, action, reward, next_state, done):
        # Store transition in replay buffer
        self.memory.push(state, action, next_state, reward, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)
        
        self.episode_step += 1
        if done:
            # Reset episode step counter
            self.episode_step = 0
    
    def act(self, state, eval_mode=False):
        state = torch.from_numpy(state).float().unsqueeze(0)
        
        # Epsilon-greedy action selection
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state)
        self.policy_net.train()
        
        # Greedy action selection
        return action_values.cpu().data.numpy().argmax()
    
    def learn(self, experiences):
        # Unpack experiences
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float()
        
        # Double DQN implementation
        # Get argmax actions from policy network
        with torch.no_grad():
            # Get the actions that would be selected by the policy network
            policy_next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            
            # Get the Q-values for these actions from the target network
            Q_targets_next = self.target_net(next_states).gather(1, policy_next_actions)
        
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from policy model
        Q_expected = self.policy_net(states).gather(1, actions)
        
        # Compute loss
        loss = F.smooth_l1_loss(Q_expected, Q_targets)  # Using Huber loss instead of MSE
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update learning rate - moved after optimizer step to fix warning
        self.scheduler.step()
        
        # Update target network
        self.soft_update()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def soft_update(self):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
    
    def save(self, filename):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        
        # Load optimizer and scheduler if they exist in the checkpoint
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'epsilon' in checkpoint:
            self.epsilon = checkpoint['epsilon'] 