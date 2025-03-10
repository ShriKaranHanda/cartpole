import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
import argparse
import os
import time
from datetime import timedelta
from collections import deque

def train_dqn(env_name='CartPole-v1', num_episodes=2000, max_t=500, 
              print_every=100, goal_score=475.0, consecutive_solves=100,
              save_checkpoint_every=500, resume_from=None, starting_episode=1):
    """Train DQN agent on specified environment.
    
    Args:
        env_name (str): Gym environment name
        num_episodes (int): Maximum number of training episodes
        max_t (int): Maximum number of timesteps per episode (500 for CartPole-v1)
        print_every (int): Print stats every this many episodes
        goal_score (float): Goal score (475.0 for CartPole-v1)
        consecutive_solves (int): Number of consecutive episodes to average for solving
        save_checkpoint_every (int): Save checkpoint models every this many episodes
        resume_from (str): Path to a model file to resume training from
        starting_episode (int): Episode number to start counting from when resuming
    """
    # Start time tracking
    start_time = time.time()
    
    env = gym.make(env_name)
    
    # Get environment information
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # Resume from checkpoint if specified
    if resume_from and os.path.isfile(resume_from):
        print(f"Resuming training from {resume_from} at episode {starting_episode}")
        agent.load(resume_from)
    
    # Training loop
    scores = []
    scores_window = []  # last consecutive_solves scores
    scores_windows = {
        "last_10": deque(maxlen=10),
        "last_50": deque(maxlen=50),
        "last_100": deque(maxlen=100)
    }
    
    best_avg_score = -float('inf')
    best_avg_score_episode = 0
    
    # Create directory for models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    for i_episode in range(starting_episode, starting_episode + num_episodes):
        episode_start_time = time.time()
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Enhanced reward shaping based on progress
            modified_reward = reward
            
            # Penalize early termination more severely
            if done and t < max_t - 1:
                # Scale penalty based on how early the failure occurs
                time_factor = 1.0 - (t / max_t)
                modified_reward = -1.0 - time_factor  # Higher penalty for earlier failures
            
            # Small bonus for lasting longer (especially in early training)
            elif t > 100 and not done:
                modified_reward = reward * 1.05
                
            # Process step
            agent.step(state, action, modified_reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward  # Still use original reward for scoring
            
            if done:
                break
                
        # Save score
        scores.append(score)
        scores_window.append(score)
        
        # Update different window sizes
        for window in scores_windows.values():
            window.append(score)
        
        # Only keep the most recent consecutive_solves scores
        if len(scores_window) > consecutive_solves:
            scores_window.pop(0)
        
        # Calculate current average score
        avg_score = np.mean(scores_window)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        # Calculate episode time
        episode_time = time.time() - episode_start_time
        
        # Save checkpoints periodically to prevent losing progress
        if i_episode % save_checkpoint_every == 0:
            checkpoint_path = f'models/{env_name}_dqn_checkpoint_{i_episode}.pth'
            agent.save(checkpoint_path)
            print(f'Checkpoint saved at episode {i_episode}')
        
        # Print progress and save best model every print_every episodes
        if i_episode % print_every == 0:
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(f'Episode {i_episode}\tScore: {score:.2f}\tAvg Score: {avg_score:.2f}\tEpsilon: {agent.epsilon:.2f}\tLR: {current_lr:.6f}\tTime: {elapsed_str}\tEps Time: {episode_time:.3f}s')
            
            # Save model if we have a new best average score
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_avg_score_episode = i_episode
                agent.save(f'models/{env_name}_dqn_best.pth')
                print(f'New best model saved with average score: {best_avg_score:.2f}')
                
                # Save additional models at various thresholds
                thresholds = [300, 350, 400, 450, 470]
                for threshold in thresholds:
                    if best_avg_score >= threshold and not os.path.exists(f'models/{env_name}_dqn_{threshold}.pth'):
                        agent.save(f'models/{env_name}_dqn_{threshold}.pth')
                        print(f'Threshold model saved at score {threshold}')
        
        # Check if environment solved
        if avg_score >= goal_score and len(scores_window) >= consecutive_solves:
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {avg_score:.2f}\tTotal Time: {elapsed_str}')
            
            # Save the final trained model
            agent.save(f'models/{env_name}_dqn_solved.pth')
            break
        
        # Early stopping condition - if almost solved, don't continue indefinitely
        elif i_episode > starting_episode + num_episodes // 2 and best_avg_score > goal_score * 0.95:
            short_window_avg = np.mean(scores_windows["last_50"])
            # If recent performance is degrading compared to best, stop early
            if short_window_avg < best_avg_score * 0.9:
                print(f'\nEarly stopping at episode {i_episode}. Recent avg {short_window_avg:.2f} declined from best {best_avg_score:.2f}')
                break
    
    # If we reach max episodes without solving, save the final model
    if i_episode >= starting_episode + num_episodes - 1:
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        print(f'\nReached max episodes without solving. Total Time: {elapsed_str}')
        print(f'Best average score: {best_avg_score:.2f} at episode {best_avg_score_episode}')
        print(f'Final average score: {avg_score:.2f}')
        
        # Save the final trained model
        agent.save(f'models/{env_name}_dqn_final.pth')
    
    return scores, agent, best_avg_score_episode

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

def test_agent(agent, env_name='CartPole-v1', n_episodes=10, max_t=500):
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
    parser.add_argument('--model_path', type=str, default='models/CartPole-v1_dqn_best.pth', help='Path to model to load for testing')
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
    parser.add_argument('--resume_from', type=str, default='models/CartPole-v1_dqn_best.pth', help='Path to model to resume training from')
    parser.add_argument('--start_episode', type=int, default=1, help='Episode to start from when resuming')
    parser.add_argument('--num_episodes', type=int, default=2000, help='Number of episodes to train for')
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    if args.train:
        if args.resume:
            # Check if model file exists
            if not os.path.isfile(args.resume_from):
                print(f"Model file '{args.resume_from}' not found.")
                print("Please specify a valid model path.")
                exit(1)
            
            scores, agent, best_episode = train_dqn(
                resume_from=args.resume_from, 
                starting_episode=args.start_episode,
                num_episodes=args.num_episodes
            )
        else:
            scores, agent, best_episode = train_dqn(num_episodes=args.num_episodes)
        
        plot_scores(scores)
        print(f"For testing, use the best model: python train.py --test --model_path models/CartPole-v1_dqn_best.pth")
        
    if args.test:
        # Check if model file exists
        if not os.path.isfile(args.model_path):
            print(f"Model file '{args.model_path}' not found.")
            print("Please train the agent first with --train flag or specify correct model path.")
            exit(1)
            
        # Create environment to get state/action sizes
        env = gym.make('CartPole-v1')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # Create agent and load trained model
        agent = DQNAgent(state_size, action_size)
        agent.load(args.model_path)
        
        # Test agent
        test_agent(agent) 