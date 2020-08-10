import os

from ray import tune
from ray.tune.schedulers import MedianStoppingRule
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import CLIReporter

from some_model_to_train import SomeModelToTrain

MODEL_FILENAME = "checkpoint.pth"
TUNE_RESULTS_FOLDER = './ray_results/'
MAX_TRAINING_ITERATION = 2000

reporter = CLIReporter(max_progress_rows=10)
reporter.add_metric_column("mean_reward")

class Trainable(tune.Trainable):
    def setup(self, hyperparameter):
        self.someModelToTrain = SomeModelToTrain(hyperparameter)

    def step(self):
        mean_reward = self.someModelToTrain.train_one_episode()
        return {'mean_reward': mean_reward}

    def save_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        self.someModelToTrain.save(checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        self.someModelToTrain.load(checkpoint_path)

# we got configuration of this from example given in: https://docs.ray.io/en/master/tune/tutorials/tune-tutorial.html
def main():
    space= {
            "batch_size": hp.choice("batch_size", [64, 128, 256]),
            "learning_rate": hp.choice("learning_rate", [0.01, 0.001, 0.0005]),
            "target_update": hp.choice("target_update", [4, 10, 100]),
            }

    hyperopt_search = HyperOptSearch(space, metric="mean_reward", mode="max")
    analysis = tune.run(
        Trainable,
        stop= {'training_iteration': MAX_TRAINING_ITERATION},
        num_samples = 10,
        scheduler=MedianStoppingRule(metric="mean_reward", mode="max"),
        search_alg=hyperopt_search,
        local_dir=TUNE_RESULTS_FOLDER,
        progress_reporter=reporter,
        checkpoint_freq=1,
        verbose=1
    )

if __name__ == "__main__":
    main()