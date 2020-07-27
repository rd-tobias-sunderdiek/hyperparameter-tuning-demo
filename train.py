import os

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch

from car import Car

MODEL_FILENAME = "saved_model"
MODEL_SAVED_IN_PATH_TXT = 'best_model_saved_in_path.txt'

class Trainable(tune.Trainable):
    def _setup(self, hyperparameter):
        self.hyperparameter = hyperparameter

    def _train(self):
        mean_reward = Car(MODEL_FILENAME).train(self.hyperparameter)
        return {'mean_reward': mean_reward}

    def _save(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        self.model.save(checkpoint_path)
        return tmp_checkpoint_dir

    def _restore(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        self.model = Car(MODEL_FILENAME).load(checkpoint_path)

# we got configuration of this from example given in: https://docs.ray.io/en/master/tune/tutorials/tune-tutorial.html
def main():
    space= {
            "actor_learning_rate": hp.choice("actor_learning_rate", [1e-4, 1e-1]),
            "critic_learning_rate": hp.choice("critic_learning_rate", [1e-4, 1e-2]),
            "tau": hp.choice("tau", [0.005, 0.1]),
            "batch_size": hp.choice("batch_size", [128, 64]),
            "action_noise_sigma": hp.choice("action_noise_sigma", [0.4, 1.0])
            }

    hyperopt_search = HyperOptSearch(space, metric="mean_reward", mode="max")
    analysis = tune.run(
        Trainable,
        stop={"training_iteration": 1},
        num_samples = 10,
        scheduler=ASHAScheduler(metric="mean_reward", mode="max"),
        search_alg=hyperopt_search,
        local_dir='./ray_results/'
    )
    print("Best hyperparameter {}".format(analysis.get_best_config(metric="mean_reward", mode="max")))
    best_model_path = analysis.get_best_logdir(metric="mean_reward", mode="max")
    print("Best model stored in {}".format(best_model_path))
    with open(MODEL_SAVED_IN_PATH_TXT, 'w') as file:
        file.write(best_model_path)

if __name__ == "__main__":
    main()