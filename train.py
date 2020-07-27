import os

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch

from car import Car

MODEL_FILENAME = "saved_model"

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
    space={"action_noise_sigma": hp.uniform("action_noise_sigma", 0.2, 0.7),
            "tau": hp.uniform("tau", 0.001, 0.01),
            "batch_size": hp.choice("batch_size", [64, 128, 256, 512]),
            "actor_learning_rate": hp.choice("actor_learning_rate", [1e-1, 1e-2, 1e-3, 1e-4]),
            "critic_learning_rate": hp.choice("critic_learning_rate", [1e-1, 1e-2, 1e-3, 1e-4])
            }

    hyperopt_search = HyperOptSearch(space, metric="mean_reward", mode="max")
    analysis = tune.run(
        Trainable,
        stop={"training_iteration": 1},
        num_samples = 50,
        scheduler=ASHAScheduler(metric="mean_reward", mode="max"),
        search_alg=hyperopt_search,
        local_dir='./ray_results/'
    )
    print("Best hyperparameter {}".format(analysis.get_best_config(metric="mean_reward", mode="max")))
    best_model_path = analysis.get_best_logdir(metric="mean_reward", mode="max")
    print("Best model stored in {}".format(best_model_path))

if __name__ == "__main__":
    main()