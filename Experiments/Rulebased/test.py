from data import IterableStatesCollection, AGENT_CLASSES
from time import time

for i in [100, int(1e3), int(1e4), int(1e5), int(1e6)]:

    trainloader = IterableStatesCollection(AGENT_CLASSES,
                                               num_players=3,
                                               agent_id=None,
                                               batch_size=8,
                                               len_iter=i)

    t0 = time()
    for data in trainloader:
        x, y = data
    print(f'Took {time() - t0} seconds with iter_len {i}')