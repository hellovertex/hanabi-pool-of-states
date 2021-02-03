import ray.tune as tune

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
pool_size = int(100) if DEBUG else int(2e3)

ray_config = {'lr': tune.loguniform(1e-4, 5e-3),  # learning rate seems to be best in [2e-3, 4e-3], old [1e-4, 1e-1]
                'num_hidden_layers': 1,  # tune.grid_search([1, 2]),
                # 'layer_size': tune.grid_search([64, 96, 128, 196, 256, 376, 448, 512]),
                # 'layer_size': tune.grid_search([64, 96, 128, 196, 256]),
                # 'layer_size': tune.grid_search([64, 128, 256]),
                'layer_size': 64,
                'batch_size': 4,  # tune.choice([4, 8, 16, 32]),
                  }

log_interval = 10 if DEBUG else 100
eval_interval = 20 if DEBUG else 500
n_states_for_evaluation = 500
""" RAY SPECIFICS """
GRACE_PERIOD = 1 if DEBUG else int(5e4)  # min num train iters before discarding bad model
MAX_T = 100 if DEBUG else int(1e5)
RAY_NUM_SAMPLES = 1 if DEBUG else 10
KEEP_CHECKPOINTS_NUM = 50
VERBOSE = 1