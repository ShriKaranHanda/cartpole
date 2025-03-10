# CartPole-DQN

A Deep Q-Network (DQN) implementation to solve the CartPole problem from OpenAI Gymnasium.

## About CartPole

CartPole is a classic control problem in reinforcement learning. The goal is to balance a pole attached to a movable cart by applying forces to the left or right. The environment is considered solved when the agent achieves an average score of 475 over 100 consecutive episodes in CartPole-v1.

State space: 4-dimensional (cart position, cart velocity, pole angle, pole angular velocity)
Action space: 2 actions (push cart left or right)
Reward: +1 for each timestep the pole remains upright
Episode termination: pole angle > 15°, cart position > 2.4, or episode length > 500

## Implementation

This implementation uses:
- **Deep Q-Network (DQN)**: Combines Q-learning with a neural network to approximate the Q-function
- **Experience Replay**: Stores and samples past experiences to break correlations
- **Target Network**: Uses a separate network for generating targets to improve stability

## Requirements

```
gymnasium==0.28.1
torch==2.0.1
numpy==1.24.3
matplotlib==3.7.1
```

## Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the agent:
```bash
python train.py --train
```
This will create a model file in the `models/` directory.

3. Test the trained agent:
```bash
python train.py --test
```
Note: You must train the agent first before testing, or specify a path to a pre-trained model:
```bash
python train.py --test --model_path models/CartPole-v1_dqn.pth
```

## Project Structure

- `dqn_agent.py`: Contains the DQN agent implementation, neural network, and replay buffer
- `train.py`: Script to train the agent on the CartPole environment and visualize results
- `models/`: Directory where trained models are saved
- `plots/`: Directory where training plots are saved

## Results

After training, the agent should be able to balance the pole for the maximum episode length of 500 timesteps consistently. Training typically takes a few hundred episodes to reach the solving criteria.

## Customization

You can modify hyperparameters such as:
- Learning rate
- Network architecture
- Epsilon decay strategy
- Replay buffer size
- Batch size
- Target network update frequency 