import os.path
import gym
import imageio
import numpy as np
import torch

from train import MODEL_SAVED_IN_PATH_TXT, MODEL_FILENAME
from pygifsicle import optimize

from some_model_to_train import SomeModelToTrain

def main():
    if (os.path.isfile(MODEL_SAVED_IN_PATH_TXT)):
        with open(MODEL_SAVED_IN_PATH_TXT, 'r') as file:
            best_model_path = file.readlines()[0]
            print("Best model found in {}, start rendering .gif".format(best_model_path))
            best_model = SomeModelToTrain()
            best_model.load(best_model_path + '/' + MODEL_FILENAME)

            # we got this part from https://stable-baselines.readthedocs.io/en/master/guide/examples.html and modified it
            env = gym.make('CartPole-v0').unwrapped
            images = []
            env.reset()
            last_screen = best_model.get_screen()
            current_screen = best_model.get_screen()
            state = current_screen - last_screen
            for i in range(75):
                action = best_model.select_action(state)
                env.step(action.item())
                img = env.render(mode='rgb_array')
                images.append(img)
                last_screen = current_screen
                current_screen = best_model.get_screen()
                next_state = current_screen - last_screen
                state = next_state
            env.close()

            imageio.mimsave('best_model.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
            optimize('best_model.gif')
    else:
        print("File {} not found".format(MODEL_SAVED_IN_PATH_TXT))

if __name__ == "__main__":
    main()