import numpy as np
players = 3
hanabi_config = {
    "colors":
      3,
    "ranks":
      5,
    "players":
      players,
    "hand_size":
      4 if players in [4, 5] else 5,
    # hand size is derived from number of players
    "max_information_tokens":
      8,
    "max_life_tokens":
      3,
    "observation_type":
      1  # pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
  }

DEBUG = True
database_size = int(5e3) if DEBUG else int(5e5)
pool_size = int(2e3)
n_states_for_evaluation = 500
max_train_steps = 1000

num_hidden_layers = 1
layer_sizes = [64, 96]

hyperparam_grid = {'lr': [0.10, 0.02],
                   'batch_size': 4}

log_interval = 10 if DEBUG else 100
eval_interval = 20 if DEBUG else 500
ckpt_interval = 500 if DEBUG else 5000