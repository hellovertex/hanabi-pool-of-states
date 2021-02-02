import tensorflow as tf
import numpy as np
import time
N_AGENTS = 2

def load_default_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(2)
    ])

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    return model


def train_data_from_replay_dict(replay_dictionary):
    X = list()
    Y = list()

    ts = time.time()
    for label, (agent_name, replays) in enumerate(replay_dictionary.items()):
        states = replays['states']
        actions = replays['actions']
        # zip here and append to X
        y = [0 for _ in range(N_AGENTS)]
        y[label] = 1
        for i_game in range(len(states)):
            for i_turn in range(len(states[i_game])):
                x = states[i_game][i_turn] + actions[i_game][i_turn]
                X.append(x)
                Y.append(y)

    # print(len(X))
    # print(len(Y))
    # print(sum(Y))
    print(f'took {time.time() - ts} seconds')


    return X, Y


# todo 1: visualize pairwise training and test accuracies
# todo 2: increase the number of different agent classes for classification
# todo 3: visualize the test accuracy as a function of the number of agents
# todo 4: apply Relevance Propagation methods, to find out which are informative features
# todo 5: visualize these heatmaps using horizontal encoding
# todo 6: apply Meta Learning techniques
# todo 7: hand in thesis