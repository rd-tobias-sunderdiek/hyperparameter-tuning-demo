import os.path
import gym
import imageio
import numpy as np

from train import MODEL_SAVED_IN_PATH_TXT, MODEL_FILENAME
from stable_baselines import DDPG
from pygifsicle import optimize

def main():
    if (os.path.isfile(MODEL_SAVED_IN_PATH_TXT)):
        with open(MODEL_SAVED_IN_PATH_TXT, 'r') as file:
            best_model_path = file.readlines()[0]
            print("Best model found in {}, start rendering .gif".format(best_model_path))
            best_model = DDPG.load(best_model_path + '/' + MODEL_FILENAME)

            # we got this part from https://stable-baselines.readthedocs.io/en/master/guide/examples.html
            env = gym.make('MountainCarContinuous-v0')
            images = []
            obs = env.reset()
            img = env.render(mode='rgb_array')
            for i in range(75):
                images.append(img)
                action, _ = best_model.predict(obs)
                obs, _, _ ,_ = env.step(action)
                img = env.render(mode='rgb_array')

            imageio.mimsave('best_model.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
            optimize('best_model.gif')
    else:
        print("File {} not found".format(MODEL_SAVED_IN_PATH_TXT))

if __name__ == "__main__":
    main()