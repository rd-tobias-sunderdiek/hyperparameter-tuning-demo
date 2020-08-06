import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG
from stable_baselines.common.callbacks import BaseCallback

class SomeModelToTrain:
    def __init__(self, hyperparameter, model_filename):
        self.model_filename = model_filename
        self.total_timesteps=200
        buffer_size = 50_000
        # we got the DDPG-example from here: https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html
        env = gym.make('MountainCarContinuous-v0')
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=hyperparameter['action_noise_sigma'] * np.ones(n_actions))
        # we got the callback example from here: https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html
        self.reward_callback = RewardCallback()
        self.model = DDPG(MlpPolicy,
                     env,
                     verbose=0,
                     action_noise=action_noise,
                     tau=hyperparameter['tau'],
                     batch_size=hyperparameter['batch_size'],
                     actor_lr=hyperparameter['actor_learning_rate'],
                     critic_lr=hyperparameter['critic_learning_rate'],
                     buffer_size=buffer_size)

    def train(self):
        self.model = self.model.learn(self.total_timesteps, callback=self.reward_callback, reset_num_timesteps=False)
        return {'mean_reward': self.reward_callback.mean_reward, 'episode_reward': self.reward_callback.episode_reward}

    def load(self, path):
        return DDPG.load(path)

    def save(self, path):
        self.model.save(self.model_filename)

# for callbacks see: https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.mean_reward = 0
        self.episode_reward = 0

    def _on_step(self) -> bool:
        self.episode_reward = self.locals['episode_reward']
        if(len(self.locals['episode_rewards_history']) == 100):
            reward_over_last_100 = np.mean(self.locals['episode_rewards_history'])
            self.mean_reward = reward_over_last_100
        return True