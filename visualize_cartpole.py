import gymnasium as gym
import torch
import numpy as np
import time
import argparse
from cartpole_rl import DQNetwork, log

# Script name for logging
SCRIPT_NAME = "visualize"

# Wrapper for the log function to include the script name
def viz_log(message):
    log(message, script_name=SCRIPT_NAME)

def visualize_agent(model_path, num_episodes=5, seed=None):
    """
    Visualize the trained agent's performance in the CartPole environment.
    
    Args:
        model_path (str): Path to the saved model file.
        num_episodes (int): Number of episodes to run.
        seed (int, optional): Random seed for reproducibility. If None, a random seed is used.
    """
    # Set up environment
    if seed is None:
        seed = int(time.time()) % 10000
    
    viz_log(f"Setting up visualization environment with seed {seed}")
    env = gym.make("CartPole-v1", render_mode="human")
    env.reset(seed=seed)
    
    # Load model
    viz_log(f"Loading model from {model_path}")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = DQNetwork(state_size, action_size, hidden_size=128)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        
        viz_log(f"Starting visualization episode {episode+1}/{num_episodes}")
        
        while not done:
            # Select action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = model(state_tensor)
            action = torch.argmax(action_values).item()
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            total_reward += reward
            
            # Add slight delay to visualize better
            time.sleep(0.01)
        
        total_rewards.append(total_reward)
        viz_log(f"Episode {episode+1} finished with reward {total_reward}")
    
    avg_reward = np.mean(total_rewards)
    viz_log(f"Visualization completed! Average reward over {num_episodes} episodes: {avg_reward:.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize the trained CartPole agent')
    parser.add_argument('--model', type=str, default="cartpole_dqn_model.pth", help='Path to the trained model file')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    visualize_agent(
        model_path=args.model, 
        num_episodes=args.episodes, 
        seed=args.seed
    ) 