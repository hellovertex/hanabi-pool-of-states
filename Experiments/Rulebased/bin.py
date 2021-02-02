import torch
from torch import nn
import ray
import os
from ray import tune
from functools import partial
import numpy as np

# todo implement like here:
#  https://discuss.pytorch.org/t/how-to-create-mlp-model-with-arbitrary-number-of-hidden-layers/13124/3

# observation
# reward_since_last_action = [0,0,0]
# reward_since_last_action = [0,1,1]
# reward_since_last_action = [1,0,2]
# but added to player 0s, 1s transitions was 1, 2

class Dynamo(nn.Module):
    def __init__(self, *layers):
        super(Dynamo, self).__init__()
        self.layers = layers

    def forward(self, x):
        return x

# todo: then 1. sample num_layers and 2. sample sizes of length 'num_layers' as follows:
# sample number of hidden layers from [1,2,3]
# sample sizes from [64 to 1024]
# sample halve from [True, False] -> if true, subsequent layers size will be halved
# sample learning rate
# sample batch_size from [4,8,16,32,64]

from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter


def load_data(data_dir):
    return torch.randint(0, 10, (2, 2))


data_dir = "."


def train(data_dir):
    data = []
    for i in range(10000):
        data.append(i)
    return i

# todo: config -> train -> tune.run -> (*) result -> config -> model
# todo where (*) inside train the metrics get reported to tune, s.t. tune.get_best_trial works
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")
    load_data(data_dir)
    # config
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    # scheduler
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2
    )
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"]
    )
    result = tune.run(
        partial(train, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
