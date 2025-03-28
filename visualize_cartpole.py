import gymnasium as gym
import torch
import numpy as np
import time
import argparse
import os
from cartpole_rl import DQNetwork, log
from torch.utils.tensorboard import SummaryWriter

# Script name for logging
SCRIPT_NAME = "visualize"

# Wrapper for the log function to include the script name
def viz_log(message):
    log(message, script_name=SCRIPT_NAME)

def visualize_agent(model_path, num_episodes=5, seed=None, use_tensorboard=True, render_mode="human"):
    """
    Visualize the trained agent's performance in the CartPole environment.
    
    Args:
        model_path (str): Path to the saved model file.
        num_episodes (int): Number of episodes to run.
        seed (int, optional): Random seed for reproducibility. If None, a random seed is used.
        use_tensorboard (bool): Whether to log visualizations to TensorBoard.
        render_mode (str): Rendering mode for gymnasium ("human" or "rgb_array").
    """
    # Set up environment
    if seed is None:
        seed = int(time.time()) % 10000
    
    viz_log(f"Setting up visualization environment with seed {seed}")
    env = gym.make("CartPole-v1", render_mode=render_mode)
    env.reset(seed=seed)
    
    # Load model
    viz_log(f"Loading model from {model_path}")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = DQNetwork(state_size, action_size, hidden_size=128)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Set up TensorBoard logging if enabled
    writer = None
    if use_tensorboard:
        if not os.path.exists("runs"):
            os.makedirs("runs")
            viz_log("Created directory for TensorBoard logs at 'runs/'")
        
        run_name = f"cartpole_visualize_seed{seed}"
        writer = SummaryWriter(f"runs/{run_name}")
        viz_log(f"TensorBoard logs will be saved to runs/{run_name}")
        
        # Log visualization hyperparameters
        hparam_dict = {
            "vis/num_episodes": num_episodes,
            "vis/seed": seed,
            "vis/model_path": model_path,
            "vis/render_mode": render_mode,
            "vis/hidden_size": 128,  # Hardcoded in this implementation
        }
        
        # We'll fill these metrics after visualization
        metric_dict = {}
        
        # Log model architecture if possible
        try:
            dummy_input = torch.zeros((1, state_size))
            writer.add_graph(model, dummy_input)
        except Exception as e:
            viz_log(f"Couldn't add model graph to TensorBoard: {e}")
    
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        states = []
        actions = []
        rewards = []
        step_count = 0
        
        viz_log(f"Starting visualization episode {episode+1}/{num_episodes}")
        
        while not done:
            # Record state for visualization
            states.append(state)
            
            # Select action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = model(state_tensor)
            
            # Log Q-values for this state
            if writer is not None and len(states) % 10 == 0:  # Log every 10 steps to reduce volume
                for a in range(action_size):
                    writer.add_scalar(f"Q_values/action_{a}", action_values[0][a].item(), 
                                     episode * 1000 + len(states))
            
            action = torch.argmax(action_values).item()
            actions.append(action)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            total_reward += reward
            step_count += 1
            
            # Add slight delay to visualize better
            time.sleep(0.01)
        
        total_rewards.append(total_reward)
        episode_lengths.append(step_count)
        viz_log(f"Episode {episode+1} finished with reward {total_reward}")
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Visualization/Episode_Reward", total_reward, episode)
            writer.add_scalar("Visualization/Episode_Length", step_count, episode)
            
            # Log state distributions
            if len(states) > 0:
                states_array = np.array(states)
                for i in range(state_size):
                    writer.add_histogram(f"State_Distribution/state_{i}", states_array[:, i], episode)
                
                # Log action distribution
                writer.add_histogram("Action_Distribution", np.array(actions), episode)
    
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    viz_log(f"Visualization completed! Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    
    # Log final metrics
    if writer is not None:
        # Create metrics for visualization run
        metric_dict = {
            "hparam/avg_reward": avg_reward,
            "hparam/min_reward": min(total_rewards),
            "hparam/max_reward": max(total_rewards),
            "hparam/avg_episode_length": avg_length, 
            "hparam/success_rate": sum(1 for r in total_rewards if r >= 500) / num_episodes
        }
        
        # Log all metrics 
        for metric_name, metric_value in metric_dict.items():
            writer.add_scalar(metric_name, metric_value, 0)
        
        # Standard metrics
        writer.add_scalar("Visualization/Avg_Reward", avg_reward, 0)
        writer.add_scalar("Visualization/Min_Reward", min(total_rewards), 0)
        writer.add_scalar("Visualization/Max_Reward", max(total_rewards), 0)
        
        # Log hyperparameters and associated metrics for the hparams dashboard
        writer.add_hparams(hparam_dict, metric_dict)
        
        writer.close()
        viz_log(f"Visualization data logged to TensorBoard")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize the trained CartPole agent')
    parser.add_argument('--model', type=str, default="cartpole_dqn_model.pth", help='Path to the trained model file')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--no-tensorboard', action='store_true', help='Disable TensorBoard logging')
    parser.add_argument('--render-mode', type=str, default="human", choices=["human", "rgb_array"], help='Rendering mode')
    args = parser.parse_args()
    
    visualize_agent(
        model_path=args.model, 
        num_episodes=args.episodes, 
        seed=args.seed,
        use_tensorboard=not args.no_tensorboard,
        render_mode=args.render_mode
    )
    
    print("\nTo view visualization data in TensorBoard, run:")
    print("tensorboard --logdir=runs") 