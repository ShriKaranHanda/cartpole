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
from torch.utils.tensorboard import SummaryWriter

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
LEARNING_RATE = 0.0001
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
def train_dqn(agent, env, num_episodes=500, max_steps=500, batch_size=64, success_threshold=487.5, seed=None, writer=None):
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
        writer: TensorBoard SummaryWriter for logging
    
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
    episode_steps = []
    
    # For tracking additional metrics
    total_steps = 0
    action_counts = [0] * agent.action_size
    
    # Log initial weights
    if writer is not None:
        for name, param in agent.policy_net.named_parameters():
            writer.add_histogram(f"initial_weights/{name}", param.data, 0)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        score = 0
        episode_losses = []
        episode_q_values = []
        episode_td_errors = []
        episode_actions = []
        episode_states = []
        steps = 0
        
        for t in range(max_steps):
            # Select and perform an action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get Q-values before action selection for logging
            with torch.no_grad():
                q_values = agent.policy_net(state_tensor)[0].detach().numpy()
                episode_q_values.append(q_values)
            
            action = agent.select_action(state)
            episode_actions.append(action)
            action_counts[action] += 1
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition in memory
            agent.memory.add(state, action, reward, next_state, done)
            
            # Learn from experience
            loss = agent.learn()
            if loss > 0:
                episode_losses.append(loss)
            
            # Store state for distribution logging
            episode_states.append(state)
            
            # Calculate TD error for this step (if we have enough samples)
            if len(agent.memory) >= batch_size:
                with torch.no_grad():
                    current_q = agent.policy_net(state_tensor).gather(1, torch.tensor([[action]])).item()
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                    next_q = agent.target_net(next_state_tensor).max(1)[0].item()
                    target_q = reward + (1 - int(done)) * agent.gamma * next_q
                    td_error = abs(current_q - target_q)
                    episode_td_errors.append(td_error)
            
            state = next_state
            score += reward
            steps += 1
            total_steps += 1
            
            if done:
                break
        
        # Log results
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        scores.append(score)
        losses.append(avg_loss)
        epsilon_values.append(agent.epsilon)
        episode_steps.append(steps)
        
        # Calculate running average
        avg_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        
        # Log to TensorBoard if writer is provided
        if writer is not None:
            # Basic metrics (already being logged)
            writer.add_scalar("Score", score, episode)
            writer.add_scalar("Avg_Loss", avg_loss, episode)
            writer.add_scalar("Epsilon", agent.epsilon, episode)
            writer.add_scalar("Avg_100_Score", avg_100, episode)
            
            # New metrics
            writer.add_scalar("Episode_Length", steps, episode)
            
            # Log Q-value statistics if we have any
            if episode_q_values:
                q_values_array = np.array(episode_q_values)
                for action_idx in range(agent.action_size):
                    writer.add_scalar(f"Q_Values/Action_{action_idx}_Mean", 
                                     np.mean(q_values_array[:, action_idx]), episode)
                    writer.add_scalar(f"Q_Values/Action_{action_idx}_Max", 
                                     np.max(q_values_array[:, action_idx]), episode)
                writer.add_scalar("Q_Values/Mean", np.mean(q_values_array), episode)
                writer.add_scalar("Q_Values/Max", np.max(q_values_array), episode)
                
                # Log Q-value distributions periodically (every 10 episodes to avoid too much data)
                if episode % 10 == 0:
                    for action_idx in range(agent.action_size):
                        writer.add_histogram(f"Q_Value_Distribution/Action_{action_idx}", 
                                           q_values_array[:, action_idx], episode)
            
            # Log TD errors
            if episode_td_errors:
                writer.add_scalar("TD_Error/Mean", np.mean(episode_td_errors), episode)
                writer.add_scalar("TD_Error/Max", np.max(episode_td_errors), episode)
                
                # Log TD error distribution periodically
                if episode % 10 == 0:
                    writer.add_histogram("TD_Error_Distribution", np.array(episode_td_errors), episode)
            
            # Log action distribution for this episode
            if episode_actions:
                writer.add_histogram("Action_Distribution", np.array(episode_actions), episode)
                for action_idx in range(agent.action_size):
                    action_pct = episode_actions.count(action_idx) / len(episode_actions) if episode_actions else 0
                    writer.add_scalar(f"Action_Freq/Action_{action_idx}", action_pct, episode)
            
            # Log state distributions periodically
            if episode % 10 == 0 and episode_states:
                states_array = np.array(episode_states)
                for i in range(agent.state_size):
                    writer.add_histogram(f"State_Distribution/State_{i}", states_array[:, i], episode)
            
            # Log weights and gradients periodically
            if episode % 20 == 0:
                for name, param in agent.policy_net.named_parameters():
                    writer.add_histogram(f"weights/{name}", param.data, episode)
                    if param.grad is not None:
                        writer.add_histogram(f"gradients/{name}", param.grad, episode)
        
        if episode % 10 == 0:
            log(f"Episode {episode}: Score = {score}, Avg Loss = {avg_loss:.4f}, Epsilon = {agent.epsilon:.4f}, Avg100 = {avg_100:.2f}")
        
        # Check if the environment is solved
        if len(scores) >= 100 and avg_100 >= success_threshold:
            log(f"Environment solved in {episode} episodes with average score of {avg_100:.2f}!")
            log(f"Success threshold was {success_threshold}")
            
            # Log final metrics to TensorBoard
            if writer is not None:
                writer.add_scalar("Final_Avg_100", avg_100, 0)
                writer.add_scalar("Episodes_to_Solve", episode, 0)
                
                # Final action distribution across all episodes
                for action_idx in range(agent.action_size):
                    writer.add_scalar(f"Final_Action_Distribution/Action_{action_idx}", 
                                     action_counts[action_idx] / total_steps, 0)
            
            break
    
    # Log final training statistics
    if writer is not None:
        # Add histograms for episode-level metrics
        writer.add_histogram("Training_Stats/Episode_Scores", np.array(scores), 0)
        writer.add_histogram("Training_Stats/Episode_Lengths", np.array(episode_steps), 0)
        writer.add_histogram("Training_Stats/Episode_Losses", np.array(losses), 0)
        
        # Final weight distributions
        for name, param in agent.policy_net.named_parameters():
            writer.add_histogram(f"final_weights/{name}", param.data, 0)
    
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
    
    # Create runs directory if it doesn't exist
    if not os.path.exists("runs"):
        os.makedirs("runs")
        log("Created directory for TensorBoard logs at 'runs/'")
    
    # Create a TensorBoard writer
    run_name = f"cartpole_main_seed{args.seed if args.seed is not None else SEED}"
    writer = SummaryWriter(f"runs/{run_name}")
    log(f"TensorBoard logs will be saved to runs/{run_name}")
    
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
    
    # Log hyperparameters - more comprehensive dictionary
    hparam_dict = {
        "hidden_size": HIDDEN_SIZE,
        "buffer_size": BUFFER_SIZE,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "learning_rate": LEARNING_RATE,
        "initial_epsilon": EPSILON,
        "epsilon_decay": EPSILON_DECAY,
        "epsilon_min": EPSILON_MIN,
        "target_update": TARGET_UPDATE,
        "max_episodes": args.episodes,
        "success_threshold": args.threshold,
        "network_structure": "2-layer MLP",
        "optimizer": "Adam",
        "loss_function": "MSE",
        "seed": SEED
    }
    
    # We'll fill these metrics after training
    metric_dict = {}
    
    # Print hyperparameters for logging
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
    
    # Try to log model graph
    try:
        dummy_input = torch.zeros((1, state_size))
        writer.add_graph(agent.policy_net, dummy_input)
    except Exception as e:
        log(f"Couldn't add model graph to TensorBoard: {e}")

    log("Starting training")
    start_time = time.time()
    scores, losses, epsilons = train_dqn(
        agent=agent,
        env=env,
        num_episodes=args.episodes,
        max_steps=500,
        batch_size=BATCH_SIZE,
        success_threshold=args.threshold,
        seed=args.seed,
        writer=writer
    )
    training_time = time.time() - start_time
    log(f"Training completed in {training_time:.2f} seconds")
    
    # Log final training time
    writer.add_scalar("Training_Time", training_time, 0)
    
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
    
    # Calculate final metrics for hparams
    final_avg_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    max_score = np.max(scores)
    avg_score = np.mean(scores)
    last_10_avg = np.mean(scores[-10:])
    episodes_to_threshold = -1
    
    # Find episodes to reach threshold
    for i in range(len(scores)):
        if i >= 99:  # Need at least 100 episodes to compute avg100
            avg100 = np.mean(scores[i-99:i+1])
            if avg100 >= args.threshold:
                episodes_to_threshold = i + 1
                break
    
    # Create complete metrics dictionary for hparams visualization
    metric_dict = {
        "hparam/final_avg_100": final_avg_100,
        "hparam/max_score": max_score, 
        "hparam/avg_score": avg_score,
        "hparam/last_10_avg": last_10_avg,
        "hparam/eval_score": eval_score,
        "hparam/episodes_to_threshold": episodes_to_threshold,
        "hparam/training_time": training_time
    }
    
    # Log all the metrics for comparing different runs
    for metric_name, metric_value in metric_dict.items():
        if metric_value != -1:  # Don't log -1 values (e.g., if threshold not reached)
            writer.add_scalar(metric_name, metric_value, 0)
    
    # Log hyperparameters and associated metrics for the hparams dashboard
    writer.add_hparams(hparam_dict, metric_dict)
    
    # Close the TensorBoard writer
    writer.close()
    
    # Close the environment
    env.close()
    log("Environment closed")
    log("CartPole reinforcement learning experiment completed successfully!")
    log("To view training progress in TensorBoard, run: tensorboard --logdir=runs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DQN agent for CartPole')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES, help='Maximum number of episodes')
    parser.add_argument('--threshold', type=float, default=SUCCESS_THRESHOLD, help='Success threshold (average score over 100 episodes)')
    args = parser.parse_args()
    
    main(args)
    
    print("\nRun the following command to execute the training:")
    print("python cartpole_rl.py [--seed SEED] [--episodes EPISODES] [--threshold THRESHOLD]")
    print("\nTo view training progress in TensorBoard, run:")
    print("tensorboard --logdir=runs") 