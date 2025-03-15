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
    
    # Track consecutive successes for early stopping
    consecutive_success_count = 0
    
    for i_episode in range(starting_episode, starting_episode + num_episodes):
        state, _ = env.reset()
        score = 0
        
        # For each timestep
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience in replay buffer and learn
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break
        
        # Save scores
        scores.append(score)
        
        # Add to sliding windows
        for window in scores_windows.values():
            window.append(score)
        
        # Calculate window means
        window_means = {k: np.mean(v) if len(v) > 0 else 0.0 for k, v in scores_windows.items()}
        
        # Save best model when we reach a new high average
        if len(scores_windows["last_100"]) == 100 and window_means["last_100"] > best_avg_score:
            best_avg_score = window_means["last_100"]
            best_avg_score_episode = i_episode
            torch.save(agent.policy_net.state_dict(), f'models/{env_name}_dqn_best.pth')
            print(f'\nEpisode {i_episode}\tBest 100-episode average: {best_avg_score:.2f}')
        
        # Save checkpoint periodically
        if i_episode % save_checkpoint_every == 0:
            torch.save(agent.policy_net.state_dict(), f'models/{env_name}_dqn_checkpoint_{i_episode}.pth')
        
        # Check if environment is solved
        if len(scores_windows["last_100"]) == 100 and window_means["last_100"] >= goal_score:
            consecutive_success_count += 1
            if consecutive_success_count >= consecutive_solves:
                print(f'\nEnvironment solved in {i_episode} episodes!')
                print(f'Average Score: {window_means["last_100"]:.2f} over last 100 episodes')
                torch.save(agent.policy_net.state_dict(), f'models/{env_name}_dqn_solved.pth')
                break
        else:
            consecutive_success_count = 0
        
        # Print progress
        if i_episode % print_every == 0:
            minutes, seconds = divmod(int(time.time() - start_time), 60)
            hours, minutes = divmod(minutes, 60)
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            print(f'Episode {i_episode}\tScore: {score:.2f}\tTime: {time_str}')
            print(f'  Average Scores: 10ep: {window_means["last_10"]:.2f}, ' +
                  f'50ep: {window_means["last_50"]:.2f}, ' +
                  f'100ep: {window_means["last_100"]:.2f}')
            print(f'  Epsilon: {agent.epsilon:.4f}')
    
    # Final model save
    torch.save(agent.policy_net.state_dict(), f'models/{env_name}_dqn_final.pth')
    
    # Print final stats
    total_time = time.time() - start_time
    time_delta = timedelta(seconds=int(total_time))
    print(f'\nTotal training time: {time_delta}')
    print(f'Episodes completed: {i_episode - starting_episode + 1} of {num_episodes}')
    print(f'Best 100-episode average: {best_avg_score:.2f} (episode {best_avg_score_episode})')
    
    return scores, agent, best_avg_score_episode

def specialized_training(model_path='models/CartPole-v1_dqn_best.pth', 
                         episodes=1000, 
                         goal_score=475.0,
                         print_every=10):
    """
    Specialized training focused on pushing model to reach target score of 475
    using curriculum learning techniques and dynamic reward shaping.
    """
    print(f"Beginning specialized training to reach score of {goal_score}...")
    print(f"Loading model from: {model_path}")
    
    # Start time tracking
    start_time = time.time()
    
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Create agent and load model
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    
    # Apply learning rate reset and fine-tuning
    new_lr = agent.reset_learning_rate(0.0002)  # Reset with slightly lower rate
    agent.fine_tune()  # Apply fine-tuning
    
    print(f"Reset learning rate to {new_lr} and applied fine-tuning")
    
    # Training loop
    scores = []
    scores_window = deque(maxlen=100)
    best_avg_score = -float('inf')
    best_score = 0
    
    # Track consecutive successes
    success_count = 0
    success_threshold = 450  # Episodes with score above this count as success
    
    for i_episode in range(1, episodes+1):
        episode_start = time.time()
        state, _ = env.reset()
        score = 0
        
        # Dynamic reward terms
        last_10_avg = np.mean(list(scores_window)[-10:]) if len(scores_window) >= 10 else 0
        progress_factor = min(1.0, i_episode / 300)  # Increases over first 300 episodes
        
        # Create curriculum difficulty - start with easier cases if struggling
        if last_10_avg < 400 and len(scores_window) >= 10:
            env.reset(options={'x_threshold': 1.8 + 0.6 * progress_factor})
        
        for t in range(500):  # Max episode length for CartPole-v1
            action = agent.act(state)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Dynamic reward shaping to encourage stability
            if not done:
                # Extract state variables
                x, x_dot, theta, theta_dot = next_state
                
                # Adjust reward based on pole angle and position (centered is better)
                angle_factor = 1.0 - min(1.0, abs(theta) / 0.2)  # Normalized angle penalty
                position_factor = 1.0 - min(1.0, abs(x) / 1.0)   # Normalized position penalty
                
                # Create shaped reward
                shaped_reward = reward * (1.0 + 0.1 * angle_factor * position_factor * progress_factor)
                
                # Add small bonus for each step agent manages to keep pole balanced
                bonus = 0.01 * min(t / 100, 1.0)  # Small bonus that increases with episode length
                
                reward = shaped_reward + bonus
            else:
                # Larger penalty for early termination
                early_term_penalty = -2.0 * (1.0 - min(t / 500, 1.0))
                reward += early_term_penalty * progress_factor
            
            # Store experience and learn
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += 1  # Still count original score (timesteps survived)
            
            if done:
                break
        
        # Save score and update windows
        scores.append(score)
        scores_window.append(score)
        
        # Track best individual score
        if score > best_score:
            best_score = score
            torch.save(agent.policy_net.state_dict(), 'models/CartPole-v1_dqn_best_score.pth')
        
        # Track best average score
        if len(scores_window) == 100:
            avg_score = np.mean(scores_window)
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                torch.save(agent.policy_net.state_dict(), 'models/CartPole-v1_dqn_specialized.pth')
                
                # If we reach goal score, we're done
                if avg_score >= goal_score:
                    print(f'\nEnvironment solved with average score: {avg_score:.2f}')
                    break
        
        # Count consecutive successes
        if score > success_threshold:
            success_count += 1
            if success_count >= 5 and i_episode % 10 == 0:
                # Save a checkpoint after 5 consecutive good episodes
                torch.save(agent.policy_net.state_dict(), 
                          f'models/CartPole-v1_dqn_specialized_checkpoint_{i_episode}.pth')
        else:
            success_count = 0
        
        # Periodically reduce learning rate
        if i_episode % 100 == 0:
            agent.reset_learning_rate(agent.optimizer.param_groups[0]['lr'] * 0.95)
        
        # Print statistics
        if i_episode % print_every == 0:
            avg_100 = np.mean(scores_window) if len(scores_window) > 0 else 0
            
            # Calculate time statistics
            episode_time = time.time() - episode_start
            total_time = time.time() - start_time
            avg_time = total_time / i_episode
            est_remain = avg_time * (episodes - i_episode)
            
            print(f'Episode {i_episode}/{episodes} | Score: {score:.1f} | ' +
                  f'Avg100: {avg_100:.1f} | Best: {best_score} | Time: {episode_time:.1f}s')
            print(f'  Elapsed: {timedelta(seconds=int(total_time))} | ' +
                  f'ETA: {timedelta(seconds=int(est_remain))}')
    
    # Return path to best model
    return 'models/CartPole-v1_dqn_specialized.pth'

def plot_scores(scores, title="DQN Training"):
    """Plot scores from training run."""
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Calculate rolling average
    rolling_mean = []
    window_size = min(100, len(scores))
    for i in range(len(scores)):
        start_idx = max(0, i - window_size + 1)
        rolling_mean.append(np.mean(scores[start_idx:(i+1)]))
    
    plt.figure(figsize=(12, 6))
    plt.plot(scores, alpha=0.4, label='Score')
    plt.plot(rolling_mean, linewidth=2, label=f'{window_size}-episode Average')
    plt.axhline(y=475, color='r', linestyle='--', label='Solving Threshold')
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('plots/training_scores.png')
    plt.close()

def test_agent(agent=None, model_path=None, env_name='CartPole-v1', n_episodes=10, max_t=500):
    """Test the agent's performance."""
    env = gym.make(env_name, render_mode='human')
    
    # Create and load agent if not provided
    if agent is None and model_path is not None:
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        agent = DQNAgent(state_size, action_size)
        agent.load(model_path)
    
    if agent is None:
        raise ValueError("Either agent or model_path must be provided")
    
    scores = []
    for i in range(n_episodes):
        state, _ = env.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.act(state, eval_mode=True)  # Use greedy policy in eval mode
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            score += reward
            
            if done:
                break
        
        scores.append(score)
        print(f'Episode {i+1}\tScore: {score}')
    
    print(f'Average Score: {np.mean(scores):.2f}')
    env.close()
    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test DQN agent on CartPole')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--specialized_train', action='store_true', help='Run specialized training to fine-tune model')
    parser.add_argument('--test', action='store_true', help='Test the agent')
    parser.add_argument('--model_path', type=str, default='models/CartPole-v1_dqn_best.pth', help='Path to model to load')
    parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
    parser.add_argument('--resume_from', type=str, default='models/CartPole-v1_dqn_best.pth', help='Path to model to resume from')
    parser.add_argument('--start_episode', type=int, default=1, help='Episode to start from when resuming')
    parser.add_argument('--num_episodes', type=int, default=2000, help='Number of episodes to train for')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes for specialized training')
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    best_model_path = args.model_path
    
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
        best_model_path = f'models/CartPole-v1_dqn_best.pth'
        print(f"For testing, use the best model: python train.py --test --model_path {best_model_path}")
    
    if args.specialized_train:
        if not os.path.isfile(args.model_path):
            print(f"Model file '{args.model_path}' not found.")
            print("Please train the agent first with --train flag or specify a valid model path.")
            exit(1)
            
        best_model_path = specialized_training(
            model_path=args.model_path,
            episodes=args.episodes
        )
        print(f"Best specialized model saved at: {best_model_path}")
        
    if args.test:
        model_to_test = best_model_path
            
        # Check if model file exists
        if not os.path.isfile(model_to_test):
            print(f"Model file '{model_to_test}' not found.")
            print("Please train the agent first with --train flag or specify correct model path.")
            exit(1)
            
        print(f"Testing model: {model_to_test}")
        test_agent(model_path=model_to_test) 