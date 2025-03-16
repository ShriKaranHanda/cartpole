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
import argparse

# Set up logging
LOG_FILE = "cartpole_training_log.txt"

def log(message, script_name="cartpole_rl"):
    """Log a message to the log file and print it to stdout."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_message = f"[{timestamp}][{script_name}] {message}"
    print(log_message)
    with open(LOG_FILE, "a") as f:
        f.write(log_message + "\n")

# Add a session separator to the log file
with open(LOG_FILE, "a") as f:
    f.write(f"\n{'-' * 80}\n")
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Starting new CartPole training session\n")
    f.write(f"{'-' * 80}\n\n")

log("Setting up the environment")
# Create the CartPole environment
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
log(f"Environment created with state size {state_size} and action size {action_size}")

# Default hyperparameters
HIDDEN_SIZE = 128
BUFFER_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
TARGET_UPDATE = 10
NUM_EPISODES = 500
SUCCESS_THRESHOLD = 487.5

# Set random seeds for reproducibility - now generating a random seed by default
SEED = int(time.time()) % 10000  # Use current time as seed by default
log(f"Using random seed {SEED} (generated from current time)")
log("Note: Set a specific seed manually for reproducibility if needed")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
env.reset(seed=SEED)

# Neural Network Model for Deep Q Learning
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

log("Implementing DQN model with architecture: state_size -> 64 -> 64 -> action_size")

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def __len__(self):
        return len(self.buffer)

log("Implementing experience replay buffer for storing transitions")

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, buffer_size=10000, batch_size=64, 
                 gamma=0.99, learning_rate=0.001, epsilon=1.0, epsilon_decay=0.995, 
                 epsilon_min=0.01, update_target_every=10):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.update_target_every = update_target_every
        self.target_update_counter = 0
        
        # Main network for training
        self.policy_net = DQNetwork(state_size, action_size, hidden_size)
        # Target network for stable learning
        self.target_net = DQNetwork(state_size, action_size, hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
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

log("Implementing DQN agent with experience replay and target network")

# Training function
def train_dqn(agent, env, num_episodes=500, max_steps=500, batch_size=64, success_threshold=487.5, seed=None):
    """
    Train the DQN agent.
    
    Args:
        agent: DQN agent
        env: Gymnasium environment
        num_episodes: Maximum number of episodes to train for
        max_steps: Maximum steps per episode
        batch_size: Batch size for training
        success_threshold: Average score over 100 episodes to consider the environment solved
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (scores, losses, epsilon_values)
    """
    if seed is not None:
        log(f"Setting training seed to {seed}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        env.reset(seed=seed)
    
    scores = []
    losses = []
    epsilon_values = []
    
    for episode in range(num_episodes):
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
        
        # Log results
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        scores.append(score)
        losses.append(avg_loss)
        epsilon_values.append(agent.epsilon)
        
        if episode % 10 == 0:
            log(f"Episode {episode}: Score = {score}, Avg Loss = {avg_loss:.4f}, Epsilon = {agent.epsilon:.4f}")
        
        # Check if the environment is solved
        if len(scores) >= 100 and np.mean(scores[-100:]) >= success_threshold:
            log(f"Environment solved in {episode} episodes with average score of {np.mean(scores[-100:]):.2f}!")
            log(f"Success threshold was {success_threshold}")
            break
    
    return scores, losses, epsilon_values

# Evaluate the trained agent
def evaluate_agent(agent, env, num_episodes=10, render=False):
    log("Evaluating the trained agent")
    test_scores = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        score = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
        
        test_scores.append(score)
        log(f"Evaluation episode {episode}: Score = {score}")
    
    avg_score = np.mean(test_scores)
    log(f"Average score over {num_episodes} evaluation episodes: {avg_score:.2f}")
    return avg_score

def main(args):
    """Main function to run the training with command line arguments."""
    global SEED
    
    if args.seed is not None:
        log(f"Using command-line provided seed: {args.seed}")
        SEED = args.seed
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        env.reset(seed=SEED)
    
    log(f"Initializing DQN agent with hyperparameters:")
    log(f"  - Hidden layer size: {HIDDEN_SIZE}")
    log(f"  - Experience buffer size: {BUFFER_SIZE}")
    log(f"  - Batch size: {BATCH_SIZE}")
    log(f"  - Discount factor (gamma): {GAMMA}")
    log(f"  - Learning rate: {LEARNING_RATE}")
    log(f"  - Initial epsilon: {EPSILON}")
    log(f"  - Epsilon decay: {EPSILON_DECAY}")
    log(f"  - Minimum epsilon: {EPSILON_MIN}")
    log(f"  - Target network update frequency: {TARGET_UPDATE}")
    log(f"  - Number of episodes: {args.episodes}")
    log(f"  - Success threshold: {args.threshold}")
    
    # Initialize the agent
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=HIDDEN_SIZE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
        update_target_every=TARGET_UPDATE
    )

    log("Starting training")
    start_time = time.time()
    scores, losses, epsilons = train_dqn(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        max_steps=500,
        batch_size=BATCH_SIZE,
        success_threshold=args.threshold,
        seed=args.seed
    )
    training_time = time.time() - start_time
    log(f"Training completed in {training_time:.2f} seconds")
    
    # Save the trained model
    model_path = "cartpole_dqn_model.pth"
    torch.save(agent.policy_net.state_dict(), model_path)
    log(f"Saved trained model to {model_path}")
    
    # Plot and save the training results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(scores)
    plt.title('Episode Scores')
    plt.ylabel('Score')
    
    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title('Loss over Episodes')
    plt.ylabel('Loss')
    
    plt.subplot(3, 1, 3)
    plt.plot(epsilons)
    plt.title('Epsilon over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig("cartpole_training_results.png")
    log("Saved training results plot to cartpole_training_results.png")
    
    # Evaluate the trained agent
    eval_score = evaluate_agent(agent, env, num_episodes=10)
    
    # Close the environment
    env.close()
    log("Environment closed")
    log("CartPole reinforcement learning experiment completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN agent for CartPole')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES, help='Maximum number of episodes')
    parser.add_argument('--threshold', type=float, default=SUCCESS_THRESHOLD, help='Success threshold (average score over 100 episodes)')
    args = parser.parse_args()
    
    main(args)
    
    print("\nRun the following command to execute the training:")
    print("python cartpole_rl.py [--seed SEED] [--episodes EPISODES] [--threshold THRESHOLD]") 