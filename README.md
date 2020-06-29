### Hyperparametertuning Demo

This is a demo for hyperparameter-tuning with Tune[1] using an DDPG example from stable-baselines[2] for the OpenAI-Gym environment[3] `BipedalWalker`[4].

This demo is meant to be able to be trained on cpu locally.

![](assets/best_model.gif)
##### Goal
Optimize mean reward of last 100 episodes (`np.mean(eval_episode_rewards)`)

#### Install
- first install OpenMPI as described in[5] (not necessary if you plan to use this repo within google colab)
- second `make install`

#### Usage
- [optional] configure hyperparameter in `train.py`
- `make train` starts training
- see results in tensorboard via `make tensorboard`
- after training finished, `make gif` creates a .gif of the best model

###### TODO
- train full example
- use different algos/scheduler
- abort if reward 300 in last 100 episodes (via callback?)
- is there a more elegant way to get the mean reward instead of my version with RewardCallback?
- configure example with redis as prerequisite
- configure alternative version which uses gpu

[1] https://docs.ray.io/en/latest/tune.html

[2] https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html

[3] https://gym.openai.com/

[4] https://gym.openai.com/envs/BipedalWalker-v2/

[5] https://stable-baselines.readthedocs.io/en/master/guide/install.html#openmpi