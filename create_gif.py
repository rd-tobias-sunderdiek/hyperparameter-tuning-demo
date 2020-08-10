import os.path
import gym
import imageio
import numpy as np
import torch

from train import MODEL_FILENAME, TUNE_RESULTS_FOLDER
from pygifsicle import optimize
from ray.tune import Analysis

from some_model_to_train import SomeModelToTrain

def main():
    analysis = Analysis(TUNE_RESULTS_FOLDER)
    print("Best hyperparameter {}".format(analysis.get_best_config(metric="mean_reward", mode="max")))
    best_model_path = analysis.get_best_logdir(metric="mean_reward", mode="max")
    print("Best model found in {}, start rendering .gif".format(best_model_path))
    best_model = SomeModelToTrain({'learning_rate': 1.0, 'batch_size': 1, 'target_update': 1})
    checkpoint_path = best_model_path + '/checkpoint_250' # todo automate
    best_model.load(checkpoint_path + '/' + MODEL_FILENAME)

    # we got this part from https://stable-baselines.readthedocs.io/en/master/guide/examples.html and modified it
    env = gym.make('LunarLander-v2')
    images = []
    for i in range(3):
        state = env.reset()
        for j in range(200):
            action = best_model.agent.act(state)
            img = env.render(mode='rgb_array')
            images.append(img)
            state, reward, done, _ = env.step(action)
            if done:
                break 
    env.close()

    imageio.mimsave('best_model.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
    optimize('best_model.gif')

if __name__ == "__main__":
    main()