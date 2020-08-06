import os

from ray import tune
from ray.tune.schedulers import MedianStoppingRule
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import CLIReporter

from some_model_to_train import SomeModelToTrain

MODEL_FILENAME = "saved_model"
MODEL_SAVED_IN_PATH_TXT = 'best_model_saved_in_path.txt'

reporter = CLIReporter(max_progress_rows=10)
reporter.add_metric_column("loss") # todo necessary?

class Trainable(tune.Trainable):
    def _setup(self, hyperparameter):
        self.model = SomeModelToTrain(hyperparameter)

    def _train(self):
        loss = self.model.train_one_episode(self.training_iteration)
        return {'loss': loss}

    def _save(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        self.model.save(checkpoint_path)
        return tmp_checkpoint_dir

    def _restore(self, tmp_checkpoint_dir):
        checkpoint_path = os.path.join(tmp_checkpoint_dir, MODEL_FILENAME)
        self.model = self.model.load(checkpoint_path)

# we got configuration of this from example given in: https://docs.ray.io/en/master/tune/tutorials/tune-tutorial.html
def main():
    space= {
            "batch_size": hp.choice("batch_size", [64, 128, 256]),
            "learning_rate": hp.choice("learning_rate", [0.01, 0.001, 0.0001]),
            "target_update": hp.choice("target_update", [10, 100]),
            }

    hyperopt_search = HyperOptSearch(space, metric="loss", mode="min")
    analysis = tune.run(
        Trainable,
        stop= {'training_iteration': 100},
        num_samples = 10,
        scheduler=MedianStoppingRule(metric="loss", mode="min"),
        search_alg=hyperopt_search,
        local_dir='./ray_results/',
        progress_reporter=reporter,
        checkpoint_at_end=False
    )
    print("Best hyperparameter {}".format(analysis.get_best_config(metric="loss", mode="min")))
    best_model_path = analysis.get_best_logdir(metric="loss", mode="min")
    print("Best model stored in {}".format(best_model_path))
    with open(MODEL_SAVED_IN_PATH_TXT, 'w') as file:
        file.write(best_model_path)

if __name__ == "__main__":
    main()