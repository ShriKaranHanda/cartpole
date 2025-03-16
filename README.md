# CartPole Reinforcement Learning Solution

This project implements a Deep Q-Network (DQN) to solve the CartPole problem from the Gymnasium (formerly OpenAI Gym) environment.

## Problem Description

CartPole is a classic reinforcement learning problem where a pole is attached to a cart that moves along a frictionless track. The goal is to prevent the pole from falling over by moving the cart left or right.

## Implementation Details

- Deep Q-Network (DQN) with experience replay and target network
- Feed-forward neural network with two hidden layers
- Epsilon-greedy exploration strategy with decay
- Complete logging of training progress

## Setup

1. Create a virtual environment:
```
python -m venv venv
```

2. Activate the virtual environment:
```
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Run the training script:
```
python cartpole_rl.py
```

The script will:
1. Train the DQN agent on the CartPole environment
2. Log all training steps to `cartpole_training_log.txt`
3. Save the trained model to `cartpole_dqn_model.pth`
4. Generate training metrics plot in `cartpole_training_results.png`
5. Evaluate the trained agent

## Hyperparameters

The implementation uses the following hyperparameters:
- Hidden layer size: 128 neurons per layer
- Experience buffer size: 10,000 transitions
- Batch size: 64
- Discount factor (gamma): 0.99
- Learning rate: 0.001
- Initial epsilon: 1.0
- Epsilon decay: 0.995
- Minimum epsilon: 0.01
- Target network update frequency: Every 10 steps
- Maximum episodes: 500

These can be modified in the script to experiment with different settings.

## Results

The environment is considered solved when the agent gets an average score of at least 195.0 over 100 consecutive episodes. 