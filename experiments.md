Questions:

- Seems like the agent could achieve a score 500 for 50 consecutive episodes, but forgot afterwards. Why?
  - Learning rate was too high
- Why doesn't the loss correspond to the reward? It seems like the loss becomes really low, but the reward seems to fluctuate
  - Because a moving target is being chased.

Tried to do a few hyperparam sweeps over network size, faster epsilon decay, higher learning rate

Changing learning rate from 0.001 to 0.0001 made the difference.

Then reproduced this for different seeds.

Tried to look at various tensorboard graphs to figure out what was wrong

Trained a model with and without target networks, and both worked well