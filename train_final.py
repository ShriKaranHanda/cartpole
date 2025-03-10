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
            # Easier starting condition (more stable)
            env.unwrapped.state = np.array([0, 0, 0.01, 0], dtype=np.float32)
            state = env.unwrapped.state
        
        for t in range(500):  # CartPole-v1 has max 500 steps
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Adaptive reward shaping
            modified_reward = reward
            
            # Penalize early termination with progressive penalty
            if done and t < 500 - 1:
                early_factor = 1.0 - (t / 500)
                modified_reward = -1.0 - (early_factor * progress_factor)
            
            # Bonus for maintaining balance longer
            elif t > 200 and not done:
                # Progressive reward increase for longer episodes
                bonus = 1.0 + (t / 500) * 0.2
                modified_reward = reward * bonus
            
            # Process step
            agent.step(state, action, modified_reward, next_state, done)
            
            # Update state and score
            state = next_state
            score += reward
            
            if done:
                break
                
        # Save score
        scores.append(score)
        scores_window.append(score)
        
        # Calculate average score
        avg_score = np.mean(scores_window)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        episode_time = time.time() - episode_start
        
        # Track consecutive successes
        if score >= success_threshold:
            success_count += 1
        else:
            success_count = 0
            
        # If we have 5 consecutive successes, increase difficulty
        if success_count >= 5:
            success_threshold = min(success_threshold + 10, 475)
            success_count = 0
            print(f"Increased success threshold to {success_threshold}")
        
        # Dynamic learning rate adjustments
        if i_episode % 100 == 0:
            # Gradually reduce learning rate
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] *= 0.95
                
        # Print progress
        if i_episode % print_every == 0:
            current_lr = agent.optimizer.param_groups[0]['lr']
            print(f'Episode {i_episode}\tScore: {score:.2f}\tAvg Score: {avg_score:.2f}\tEpsilon: {agent.epsilon:.2f}\tLR: {current_lr:.6f}\tTime: {elapsed_str}\tEps Time: {episode_time:.3f}s')
        
        # Save best score
        if score > best_score:
            best_score = score
            agent.save(f'models/CartPole-v1_dqn_best_score.pth')
            print(f'New best individual score: {best_score:.2f}')
            
        # Save if we've reached a new best average
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            agent.save(f'models/CartPole-v1_dqn_specialized.pth')
            print(f'New best model saved with average score: {best_avg_score:.2f}')
            
            # Save when reaching certain thresholds
            thresholds = [450, 460, 470, 475]
            for threshold in thresholds:
                if best_avg_score >= threshold:
                    agent.save(f'models/CartPole-v1_dqn_{threshold}.pth')
                    print(f'Milestone! Saved model at {threshold} score')
        
        # Check if environment solved
        if avg_score >= goal_score:
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            print(f'\nEnvironment solved in {i_episode} episodes!\tAverage Score: {avg_score:.2f}\tTotal Time: {elapsed_str}')
            agent.save(f'models/CartPole-v1_dqn_solved.pth')
            break
    
    # Training complete
    elapsed = time.time() - start_time
    elapsed_str = str(timedelta(seconds=int(elapsed)))
    print(f'\nTraining completed. Total Time: {elapsed_str}')
    print(f'Best average score: {best_avg_score:.2f}')
    print(f'Best individual score: {best_score:.2f}')
    
    # Return the best model path
    if best_avg_score >= goal_score:
        return f'models/CartPole-v1_dqn_solved.pth'
    elif best_avg_score >= 450:
        return f'models/CartPole-v1_dqn_specialized.pth'
    else:
        return model_path

def test_agent(model_path, n_episodes=10, max_t=500):
    """Test the trained agent."""
    env = gym.make('CartPole-v1', render_mode='human')
    
    # Load agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    
    scores = []
    
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
        
        scores.append(score)
        print(f'Episode {i+1}\tScore: {score}')
    
    print(f'Average Score: {np.mean(scores):.2f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specialized training for CartPole')
    parser.add_argument('--train', action='store_true', help='Run specialized training')
    parser.add_argument('--test', action='store_true', help='Test the agent')
    parser.add_argument('--model_path', type=str, default='models/CartPole-v1_dqn_best.pth', help='Path to model file')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    if args.train:
        if not os.path.isfile(args.model_path):
            print(f"Model file '{args.model_path}' not found.")
            exit(1)
            
        best_model_path = specialized_training(
            model_path=args.model_path,
            episodes=args.episodes
        )
        print(f"Best model saved at: {best_model_path}")
        
    if args.test:
        model_to_test = args.model_path
        if args.train:
            # If we just trained, use that model
            model_to_test = best_model_path
            
        # Check if model file exists
        if not os.path.isfile(model_to_test):
            print(f"Model file '{model_to_test}' not found.")
            exit(1)
            
        print(f"Testing model: {model_to_test}")
        test_agent(model_to_test) 