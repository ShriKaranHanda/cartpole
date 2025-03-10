import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import argparse
import os

def train_dqn(env_name='CartPole-v1', num_episodes=1000, max_t=1000, 
              print_every=100, goal_score=195.0, consecutive_solves=100):
    """Train DQN agent on specified environment.
    
    Args:
        env_name (str): Gym environment name
        num_episodes (int): Maximum number of training episodes
        max_t (int): Maximum number of timesteps per episode
        print_every (int): Print stats every this many episodes
        goal_score (float): Goal score (avg over consecutive_solves)
        consecutive_solves (int): Number of consecutive episodes to average for solving
    """
    env = gym.make(env_name)
    
    # Get environment information
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Training loop
    scores = []
    scores_window = []  # last consecutive_solves scores
    
    for i_episode in range(1, num_episodes+1):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process step
            agent.step(state, action, reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            if done:
                break
                
        # Save score
        scores.append(score)
        scores_window.append(score)
        
        # Only keep the most recent consecutive_solves scores
        if len(scores_window) > consecutive_solves:
            scores_window.pop(0)
        
        # Print progress
        if i_episode % print_every == 0:
            avg_score = np.mean(scores_window)
            print(f'Episode {i_episode}\tAverage Score: {avg_score:.2f}\tEpsilon: {agent.epsilon:.2f}')
        
        # Check if environment solved
        if np.mean(scores_window) >= goal_score and len(scores_window) >= consecutive_solves:
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            
            # Save the trained model
            if not os.path.exists('models'):
                os.makedirs('models')
            agent.save(f'models/{env_name}_dqn.pth')
            break
    
    return scores, agent

def plot_scores(scores, title="DQN Training"):
    """Plot the scores."""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(np.arange(len(scores)), scores)
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    ax.set_title(title)
    
    # Save the plot
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig('plots/training_scores.png')
    plt.show()

def test_agent(agent, env_name='CartPole-v1', n_episodes=10, max_t=1000):
    """Test the trained agent."""
    env = gym.make(env_name, render_mode='human')
    
    for i in range(n_episodes):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state, eval_mode=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
            
            if done:
                break
        
        print(f'Episode {i+1}\tScore: {score}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN agent on CartPole')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--test', action='store_true', help='Test the agent')
    parser.add_argument('--model_path', type=str, default='models/CartPole-v1_dqn.pth', help='Path to model to load for testing')
    args = parser.parse_args()
    
    if args.train:
        scores, agent = train_dqn()
        plot_scores(scores)
        
    if args.test:
        # Create environment to get state/action sizes
        env = gym.make('CartPole-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # Create agent and load trained model
        agent = DQNAgent(state_size, action_size)
        agent.load(args.model_path)
        
        # Test agent
        test_agent(agent) 