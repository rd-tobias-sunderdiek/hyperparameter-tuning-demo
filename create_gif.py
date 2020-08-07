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
            best_model = SomeModelToTrain({'learning_rate': 1.0, 'batch_size': 1, 'target_update': 1})
            best_model.load(best_model_path + '/' + MODEL_FILENAME)

            # we got this part from https://stable-baselines.readthedocs.io/en/master/guide/examples.html and modified it
            env = gym.make('LunarLander-v2')
            images = []
            for i in range(3):
                state = env.reset()
                for j in range(200):
                    action = best_model.agent.act(state)
                    #env.render()
                    img = env.render(mode='rgb_array')
                    images.append(img)
                    state, reward, done, _ = env.step(action)
                    if done:
                        break 
            env.close()

            imageio.mimsave('best_model.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
            optimize('best_model.gif')
    else:
        print("File {} not found".format(MODEL_SAVED_IN_PATH_TXT))

if __name__ == "__main__":
    main()