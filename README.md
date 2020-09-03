### Hyperparametertuning Demo

This is a demo for hyperparameter-tuning with Tune[1] using an DQN example from Udacity[2] for the OpenAI-Gym environment[3] `LunarLander`[4].

This demo is meant to be able to be trained on cpu locally (took ~40 min. on a 2.5 GHz Quad-Core i7)

![best model](assets/best_model.gif)

##### Goal
Land on the moon and get reward for landing properly, loose reward for using fuel or land outside landing pad. In this example, we use the metric `mean_reward` for this.

#### Install
- `make install` (tested with python 3.8)
- install ray version 0.9.0-dev0 for your OS like described in [5]

#### Usage
- watch random, untrained agent via `make random`
- [optional] configure hyperparameter in `train.py`
- `make train` starts training
- see results in tensorboard via `make tensorboard`
- after training finished, `make gif` creates a .gif of the best model

#### TODO
- 0.9.0-dev0

[1] https://docs.ray.io/en/latest/tune.html

[2] https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution

[3] https://gym.openai.com/

[4] https://gym.openai.com/envs/LunarLander-v2/

[5] https://docs.ray.io/en/latest/installation.html
