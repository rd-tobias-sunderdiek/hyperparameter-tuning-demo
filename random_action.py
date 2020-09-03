import os.path
import gym
import imageio
import numpy as np
import random
import torch

from train import MODEL_FILENAME, TUNE_RESULTS_FOLDER, MAX_TRAINING_ITERATION
from ray.tune import Analysis

from some_model_to_train import SomeModelToTrain

def main():
    env = gym.make('LunarLander-v2')
    images = []
    state = env.reset()
    for i in range(100):
        for j in range(120):
            action = random.choice([0, 1, 2])
            img = env.render(mode='rgb_array')
            images.append(img)
            state, reward, done, _ = env.step(action)
            if done:
                env.reset()
    env.close()


if __name__ == "__main__":
    main()