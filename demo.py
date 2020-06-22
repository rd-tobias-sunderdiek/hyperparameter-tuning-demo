import gym
import numpy as np
import os

from ray import tune
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.common.callbacks import BaseCallback

# for callbacks see: https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.mean_reward = 0

    def _on_training_end(self) -> None:
        eval_episode_rewards = self.locals['eval_episode_rewards']
        self.mean_reward = np.mean(eval_episode_rewards)

    def mean_reward(self):
        return self.mean_reward

class Trainable(tune.Trainable):
    def _setup(self, config):
        self.config = config

    def _train(self):
        # we got the DDPG-example from here: https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html
        # but used BipdedalWalker instead MountainCarContinuous in our demo
        env = gym.make('BipedalWalker-v3')
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=self.config['action_noise_sigma'] * np.ones(n_actions))
        reward_callback = RewardCallback()
        model = DDPG(MlpPolicy, env, verbose=0, param_noise=param_noise, action_noise=action_noise)
        model.learn(total_timesteps, callback=reward_callback)
        return {'mean_reward': reward_callback.mean_reward}

    def _save(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "bipedal_walker_model")
        self.model.save(checkpoint_path)
        return tmp_checkpoint_dir

    def _restore(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "bipedal_walker_model")
        self.model = DDPG.load(checkpoint_path)

#TODO configure parameter
total_timesteps=4#400000
param_noise = None

analysis = tune.run(
    Trainable,
    stop={"training_iteration": 20},
    config={"action_noise_sigma": tune.grid_search([float(0.5), float(0.9)])},
    local_dir='./ray_results/'
)

print("Best hyperparameter {}".format(analysis.get_best_config(metric="mean_reward", mode="max")))