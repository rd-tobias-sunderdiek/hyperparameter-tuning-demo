### Hyperparametertuning Demo

This is a demo for hyperparameter-tuning with Tune[1] using an DDPG example from stable-baselines[2] for the OpenAI-Gym environment[3] `MountainCarContinuous`[4].

This demo is meant to be able to be trained on cpu locally.

![best model](assets/best_model.gif)

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
- use different scheduler without redis
- describe goal and rewards in readme

[1] https://docs.ray.io/en/latest/tune.html

[2] https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html

[3] https://gym.openai.com/

[4] https://gym.openai.com/envs/MountainCarContinuous-v0/

[5] https://stable-baselines.readthedocs.io/en/master/guide/install.html#openmpi