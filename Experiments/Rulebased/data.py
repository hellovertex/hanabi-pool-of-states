import torch
import numpy as np
from cl2 import StateActionCollector, AGENT_CLASSES
from typing import Dict, Optional
from time import time
from itertools import cycle, chain
import random
import rulebased_agent as ra
from internal_agent import InternalAgent
from outer_agent import OuterAgent
from iggi_agent import IGGIAgent
from legal_random_agent import LegalRandomAgent
from flawed_agent import FlawedAgent
from piers_agent import PiersAgent
from van_den_bergh_agent import VanDenBerghAgent

# AGENT_CLASSES = {'InternalAgent': InternalAgent,
#                  'OuterAgent': OuterAgent, 'IGGIAgent': IGGIAgent, 'FlawedAgent': FlawedAgent,
#                  'PiersAgent': PiersAgent, 'VanDenBerghAgent': VanDenBerghAgent}


class StateActionWriter:
    """ - Collects states and actions using the StateActionCollector
        - writes them to database
    """

    def __init__(self,
                 agent_classes: Dict[str, ra.RulebasedAgent],
                 num_players: int,
                 # max_size_per_iter: int = 100,
                 target_agent: Optional[str] = None):
        self._data_collector = StateActionCollector(agent_classes, num_players, target_agent)

    def collect_and_write_to_database(self, path_to_db, num_rows_to_add, keep_state_dict=True):
        # if path_to_db does not exists, create a file, otherwise append to database
        #        x          x       x      x       x        x
        # | num_players | agent | turn | state | action | team |
        """ If use_state_dict is True, a different table will be used, that stores the
        observation as a dictionary """
        collected = 0
        while collected < num_rows_to_add:
            self._data_collector.collect(num_states_to_collect=1000,  # only thousand at once because its slow otherwise
                                         insert_to_database_at=path_to_db,
                                         keep_obs_dict=keep_state_dict)
            collected += 1000
            print(f'Collected {collected} states and wrote them to {path_to_db}')


def collect_and_write_to_database(path_to_db, num_rows_to_add, keep_state_dict=False):
    writer = StateActionWriter(AGENT_CLASSES, 3)
    writer.collect_and_write_to_database(path_to_db, num_rows_to_add, keep_state_dict=keep_state_dict)


def collect(num_states_to_collect):
    collector = StateActionCollector(AGENT_CLASSES, 3)
    states = collector.collect(drop_actions=False,
                               num_states_to_collect=num_states_to_collect,
                               target_agent=None,
                               keep_obs_dict=True,
                               keep_agent=False)
    return states
# collect_and_write_to_database(path_to_db='./database_test.db', num_rows_to_add=int(5e5), keep_state_dict=True)
# # too slow, seconds taken increase linearly with number of states
# start = time()
# print(collect(1e2), f'took {time() - start} seconds')
# start = time()
# print(collect(1e3), f'took {time() - start} seconds')
# start = time()
# print(collect(1e4), f'took {time() - start} seconds')
#start = time()
#print(collect(1e2), f'took {time() - start} seconds')
# print(f'took {time() - start} seconds')
# print(type(data))
# print(type(data['obs_dicts']))
