import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import time
import os
from cartpole_rl import DQNetwork, ReplayBuffer, log

# Script name for logging
SCRIPT_NAME = "experiment"

# Wrapper for the log function to include the script name
def experiment_log(message):
    log(message, script_name=SCRIPT_NAME)

# Success threshold
SUCCESS_THRESHOLD = 487.5

# Generate different seeds for each experiment to ensure diversity
BASE_SEED = int(time.time()) % 10000
experiment_log(f"Base seed for experiments: {BASE_SEED}")

# Hyperparameter sets to experiment with
experiment_configs = [
    {
        "name": "Baseline",
        "hidden_size": 128,
        "buffer_size": 10000,
        "batch_size": 64,
        "gamma": 0.99,
        "learning_rate": 0.001,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "target_update": 10,
        "seed": BASE_SEED
    },
    {
        "name": "Deeper Network",
        "hidden_size": 256,
        "buffer_size": 10000,
        "batch_size": 64,
        "gamma": 0.99,
        "learning_rate": 0.001,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "target_update": 10,
        "seed": BASE_SEED + 1
    },
    {
        "name": "Faster Epsilon Decay",
        "hidden_size": 128,
        "buffer_size": 10000,
        "batch_size": 64,
        "gamma": 0.99,
        "learning_rate": 0.001,
        "epsilon": 1.0,
        "epsilon_decay": 0.98,
        "epsilon_min": 0.01,
        "target_update": 10,
        "seed": BASE_SEED + 2
    },
    {
        "name": "Higher Learning Rate",
        "hidden_size": 128,
        "buffer_size": 10000,
        "batch_size": 64,
        "gamma": 0.99,
        "learning_rate": 0.005,
        "epsilon": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        "target_update": 10,
        "seed": BASE_SEED + 3
    }
]

class DQNAgentExperiment:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.memory = ReplayBuffer(config["buffer_size"])
        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]
        self.update_target_every = config["target_update"]
        self.target_update_counter = 0
        
        # Main network for training
        self.policy_net = DQNetwork(state_size, action_size, config["hidden_size"])
        # Target network for stable learning
        self.target_net = DQNetwork(state_size, action_size, config["hidden_size"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config["learning_rate"])
        self.loss_fn = nn.MSELoss()
    
    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.policy_net(state)
        return torch.argmax(action_values).item()
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor([1 if done else 0 for done in dones])
        
        # Compute Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Compute loss and optimize
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.target_update_counter += 1
        if self.target_update_counter % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

def train_experiment(config, max_episodes=200, max_steps=500):
    """Train a DQN agent with a specific hyperparameter configuration."""
    # Set random seeds for reproducibility
    seed = config["seed"]
    experiment_log(f"Using seed {seed} for experiment: {config['name']}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create the CartPole environment
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize the agent
    agent = DQNAgentExperiment(state_size, action_size, config)
    
    # Training loop
    scores = []
    losses = []
    epsilon_values = []
    solved_episode = -1
    
    experiment_log(f"Starting experiment with configuration: {config['name']}")
    start_time = time.time()
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        score = 0
        episode_losses = []
        
        for t in range(max_steps):
            # Select and perform an action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition in memory
            agent.memory.add(state, action, reward, next_state, done)
            
            # Learn from experience
            loss = agent.learn()
            if loss > 0:
                episode_losses.append(loss)
            
            state = next_state
            score += reward
            
            if done:
                break
        
        # Record results
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        scores.append(score)
        losses.append(avg_loss)
        epsilon_values.append(agent.epsilon)
        
        if episode % 10 == 0:
            experiment_log(f"Experiment {config['name']} - Episode {episode}: Score = {score}, Avg Loss = {avg_loss:.4f}, Epsilon = {agent.epsilon:.4f}")
        
        # Check if the environment is solved - using the updated threshold
        if len(scores) >= 100 and np.mean(scores[-100:]) >= SUCCESS_THRESHOLD and solved_episode == -1:
            solved_episode = episode
            experiment_log(f"Experiment {config['name']} - Environment solved in {episode} episodes with average score of {np.mean(scores[-100:]):.2f}!")
            experiment_log(f"Success threshold was {SUCCESS_THRESHOLD}")
    
    training_time = time.time() - start_time
    experiment_log(f"Experiment {config['name']} completed in {training_time:.2f} seconds")
    
    # Close the environment
    env.close()
    
    return {
        "config_name": config["name"],
        "scores": scores,
        "losses": losses,
        "epsilon_values": epsilon_values,
        "training_time": training_time,
        "solved_episode": solved_episode,
        "final_avg_100": np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores),
        "seed": config["seed"]
    }

def run_experiments():
    experiment_log("Starting hyperparameter experiments for CartPole")
    results = []
    
    for config in experiment_configs:
        result = train_experiment(config)
        results.append(result)
    
    # Compare results
    experiment_log("\nExperiment Results Summary:")
    experiment_log("-" * 80)
    experiment_log(f"{'Configuration':<20} | {'Solved Episode':<15} | {'Training Time':<15} | {'Final Avg 100':<15}")
    experiment_log("-" * 80)
    
    for result in results:
        solved_str = str(result["solved_episode"]) if result["solved_episode"] != -1 else "Not solved"
        experiment_log(f"{result['config_name']:<20} | {solved_str:<15} | {result['training_time']:.2f}s{' ':<7} | {result['final_avg_100']:.2f}{' ':<7}")
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    for result in results:
        plt.plot(result["scores"], label=result["config_name"])
    plt.title('Comparison of Episode Scores')
    plt.ylabel('Score')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for result in results:
        plt.plot(result["losses"], label=result["config_name"])
    plt.title('Comparison of Loss over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("cartpole_experiment_results.png")
    experiment_log("Saved experiment results plot to cartpole_experiment_results.png")

if __name__ == "__main__":
    run_experiments() 