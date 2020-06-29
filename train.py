import gym
import numpy as np
import os

from ray import tune
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines import DDPG
from stable_baselines.common.callbacks import BaseCallback

MODEL_FILENAME = "bipedal_walker_model"
MODEL_SAVED_IN_PATH_TXT = 'best_model_saved_in_path.txt'

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
        self.total_timesteps=1#00000
        self.buffer_size = 50000

    def _train(self):
        # we got the DDPG-example from here: https://stable-baselines.readthedocs.io/en/master/modules/ddpg.html
        # but used BipdedalWalker instead MountainCarContinuous in our demo
        env = gym.make('BipedalWalker-v3')
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=self.config['action_noise_sigma'] * np.ones(n_actions))
        reward_callback = RewardCallback()
        model = DDPG(MlpPolicy,
                     env,
                     verbose=0,
                     action_noise=action_noise,
                     tau=self.config['tau'],
                     batch_size=self.config['batch_size'],
                     actor_lr=self.config['actor_learning_rate'],
                     critic_lr=self.config['critic_learning_rate'],
                     buffer_size=self.buffer_size)
        model.learn(self.total_timesteps, callback=reward_callback)
        model.save(MODEL_FILENAME)
        return {'mean_reward': reward_callback.mean_reward}

    def _save(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        self.model.save(checkpoint_path)
        return tmp_checkpoint_dir

    def _restore(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        self.model = DDPG.load(checkpoint_path)

def main():
    analysis = tune.run(
        Trainable,
        stop={"training_iteration": 1},
        config={"action_noise_sigma": tune.grid_search([float(0.5)]),
                "tau": tune.grid_search([float(0.001)]),
                "batch_size": tune.grid_search([256]),
                "actor_learning_rate": tune.grid_search([float(0.000527)]),
                "critic_learning_rate": tune.grid_search([float(0.001)])
                },
        local_dir='./ray_results/'
    )
    print("Best hyperparameter {}".format(analysis.get_best_config(metric="mean_reward", mode="max")))
    best_model_path = analysis.get_best_logdir(metric="mean_reward", mode="max")
    print("Best model stored in {}".format(best_model_path))
    with open(MODEL_SAVED_IN_PATH_TXT, 'w') as file:
        file.write(best_model_path)

if __name__ == "__main__":
    main()