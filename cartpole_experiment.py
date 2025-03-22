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
from torch.utils.tensorboard import SummaryWriter

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

# Create a runs directory for TensorBoard logs if it doesn't exist
if not os.path.exists("runs"):
    os.makedirs("runs")
    experiment_log("Created directory for TensorBoard logs at 'runs/'")

# Hyperparameter sets to experiment with
experiment_configs = [
    {
        "name": "Baseline",
        "hidden_size": 128,
        "buffer_size": 10000,
        "batch_size": 64,
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
        target_q_values = rewards + (next_q_values * (1 - dones))
        
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

def train_experiment(config, max_episodes=500, max_steps=500):
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
    
    # Create a TensorBoard writer for this experiment
    writer = SummaryWriter(f"runs/cartpole_{config['name']}_seed{seed}")
    
    # Log hyperparameters to TensorBoard as text
    writer.add_text("hyperparams", str(config))
    
    # Create a more comprehensive hyperparameter dictionary
    hparam_dict = {
        "hidden_size": config["hidden_size"],
        "buffer_size": config["buffer_size"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "initial_epsilon": config["epsilon"],
        "epsilon_decay": config["epsilon_decay"],
        "epsilon_min": config["epsilon_min"],
        "target_update": config["target_update"],
        "max_episodes": max_episodes,
        "success_threshold": SUCCESS_THRESHOLD,
        "experiment_name": config["name"],
        "network_structure": "2-layer MLP",
        "optimizer": "Adam",
        "loss_function": "MSE",
        "seed": seed
    }
    
    # We'll fill these metrics after training
    metric_dict = {}
    
    # Try to log model graph
    try:
        dummy_input = torch.zeros((1, state_size))
        writer.add_graph(agent.policy_net, dummy_input)
    except Exception as e:
        experiment_log(f"Couldn't add model graph to TensorBoard: {e}")
    
    # Log initial weights
    for name, param in agent.policy_net.named_parameters():
        writer.add_histogram(f"initial_weights/{name}", param.data, 0)
    
    # Training loop
    scores = []
    losses = []
    epsilon_values = []
    episode_steps = []
    solved_episode = -1
    
    # Additional tracking variables
    total_steps = 0
    action_counts = [0] * action_size
    
    experiment_log(f"Starting experiment with configuration: {config['name']}")
    start_time = time.time()
    
    for episode in range(max_episodes):
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
            if len(agent.memory) >= agent.batch_size:
                with torch.no_grad():
                    current_q = agent.policy_net(state_tensor).gather(1, torch.tensor([[action]])).item()
                    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                    next_q = agent.target_net(next_state_tensor).max(1)[0].item()
                    target_q = reward + (1 - int(done)) * next_q
                    td_error = abs(current_q - target_q)
                    episode_td_errors.append(td_error)
            
            state = next_state
            score += reward
            steps += 1
            total_steps += 1
            
            if done:
                break
        
        # Record results
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        scores.append(score)
        losses.append(avg_loss)
        epsilon_values.append(agent.epsilon)
        episode_steps.append(steps)
        
        # Calculate running average
        avg_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        
        # Log to TensorBoard
        writer.add_scalar("Score", score, episode)
        writer.add_scalar("Avg_Loss", avg_loss, episode)
        writer.add_scalar("Epsilon", agent.epsilon, episode)
        writer.add_scalar("Avg_100_Score", avg_100, episode)
        writer.add_scalar("Episode_Length", steps, episode)
        
        # Log Q-value statistics if we have any
        if episode_q_values:
            q_values_array = np.array(episode_q_values)
            for action_idx in range(action_size):
                writer.add_scalar(f"Q_Values/Action_{action_idx}_Mean", 
                                 np.mean(q_values_array[:, action_idx]), episode)
                writer.add_scalar(f"Q_Values/Action_{action_idx}_Max", 
                                 np.max(q_values_array[:, action_idx]), episode)
            writer.add_scalar("Q_Values/Mean", np.mean(q_values_array), episode)
            writer.add_scalar("Q_Values/Max", np.max(q_values_array), episode)
            
            # Log Q-value distributions periodically
            if episode % 10 == 0:
                for action_idx in range(action_size):
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
            for action_idx in range(action_size):
                action_pct = episode_actions.count(action_idx) / len(episode_actions) if episode_actions else 0
                writer.add_scalar(f"Action_Freq/Action_{action_idx}", action_pct, episode)
        
        # Log state distributions periodically
        if episode % 10 == 0 and episode_states:
            states_array = np.array(episode_states)
            for i in range(state_size):
                writer.add_histogram(f"State_Distribution/State_{i}", states_array[:, i], episode)
        
        # Log weights and gradients periodically
        if episode % 20 == 0:
            for name, param in agent.policy_net.named_parameters():
                writer.add_histogram(f"weights/{name}", param.data, episode)
                if param.grad is not None:
                    writer.add_histogram(f"gradients/{name}", param.grad, episode)
        
        if episode % 10 == 0:
            experiment_log(f"Experiment {config['name']} - Episode {episode}: Score = {score}, Avg Loss = {avg_loss:.4f}, Epsilon = {agent.epsilon:.4f}")
        
        # Check if the environment is solved - using the updated threshold
        if len(scores) >= 100 and avg_100 >= SUCCESS_THRESHOLD and solved_episode == -1:
            solved_episode = episode
            experiment_log(f"Experiment {config['name']} - Environment solved in {episode} episodes with average score of {avg_100:.2f}!")
            experiment_log(f"Success threshold was {SUCCESS_THRESHOLD}")
    
    training_time = time.time() - start_time
    experiment_log(f"Experiment {config['name']} completed in {training_time:.2f} seconds")
    
    # Calculate final metrics for hparams and tensorboard dashboard
    final_avg_100 = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    max_score = np.max(scores)
    avg_score = np.mean(scores)
    mean_episode_length = np.mean(episode_steps)
    last_10_avg = np.mean(scores[-10:])
    episodes_to_threshold = solved_episode if solved_episode != -1 else -1
    
    # Create comprehensive metrics dictionary for the hparams dashboard
    metric_dict = {
        "hparam/final_avg_100": final_avg_100,
        "hparam/max_score": max_score,
        "hparam/avg_score": avg_score,
        "hparam/last_10_avg": last_10_avg,
        "hparam/mean_episode_length": mean_episode_length,
        "hparam/episodes_to_threshold": episodes_to_threshold if episodes_to_threshold != -1 else 0,
        "hparam/training_time": training_time,
        "hparam/solved": 1 if episodes_to_threshold != -1 else 0
    }
    
    # Log all the metrics for comparing different runs
    for metric_name, metric_value in metric_dict.items():
        writer.add_scalar(metric_name, metric_value, 0)
    
    # Log standard metrics for standard dashboard
    writer.add_scalar("Final_Avg_100", final_avg_100, 0)
    writer.add_scalar("Training_Time", training_time, 0)
    
    # Log hyperparameters and associated metrics for the hparams dashboard
    writer.add_hparams(hparam_dict, metric_dict)
    
    # Log final training statistics
    writer.add_histogram("Training_Stats/Episode_Scores", np.array(scores), 0)
    writer.add_histogram("Training_Stats/Episode_Lengths", np.array(episode_steps), 0)
    writer.add_histogram("Training_Stats/Episode_Losses", np.array(losses), 0)
    
    # Final action distribution across all episodes
    for action_idx in range(action_size):
        writer.add_scalar(f"Final_Action_Distribution/Action_{action_idx}", 
                         action_counts[action_idx] / total_steps if total_steps > 0 else 0, 0)
    
    # Final weight distributions
    for name, param in agent.policy_net.named_parameters():
        writer.add_histogram(f"final_weights/{name}", param.data, 0)
    
    # Close the tensorboard writer
    writer.close()
    
    # Close the environment
    env.close()
    
    return {
        "config_name": config["name"],
        "scores": scores,
        "losses": losses,
        "epsilon_values": epsilon_values,
        "training_time": training_time,
        "solved_episode": solved_episode,
        "final_avg_100": final_avg_100,
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
    experiment_log("To view all experiments in TensorBoard, run: tensorboard --logdir=runs")

if __name__ == "__main__":
    run_experiments() 