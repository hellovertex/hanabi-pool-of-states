from typing import List, Dict, Optional, Tuple
import numpy as np
import pickle
import random
import rl_env
import os
import torch
import torchvision
import rulebased_agent as ra
from internal_agent import InternalAgent
from outer_agent import OuterAgent
from iggi_agent import IGGIAgent
from legal_random_agent import LegalRandomAgent
from flawed_agent import FlawedAgent
from piers_agent import PiersAgent
from van_den_bergh_agent import VanDenBerghAgent
import random
from typing import Optional
from collections import namedtuple
AGENT_CLASSES = {'InternalAgent': InternalAgent,
                 'OuterAgent': OuterAgent, 'IGGIAgent': IGGIAgent, 'FlawedAgent': FlawedAgent,
                 'PiersAgent': PiersAgent, 'VanDenBerghAgent': VanDenBerghAgent}

COLORS = ['R', 'Y', 'G', 'W', 'B']

# todo: IterableDataset -> StatesCollector -> Runner

replay_dictionary = {'agents': ['A', 'B', 'C'],
                     'agent_1': {'states': [[1,2,3], [4,5,6]], 'actions': [10,20]},
                     'agent_2': {'states': [[1,2,3], [4,5,6]], 'actions': [10,20]},
                     'agent_3': {'states': [[1,2,3], [4,5,6]], 'actions': [10,20]},
                     'info': None}

class IterableStatesCollector:
    def __init__(self, agent_pool: Dict[str, ra.RulebasedAgent],
                 num_players: int = 3,
                 drop_actions: bool = True,
                 checkpoint_dir: Optional[str] = None,
                 target_agent: Optional[str] = None,
                 return_lazily: bool = True):
        """
        IterableStatesCollector can be used to either
        - return a torch.utils.data.IterableDataset for training (possibly with lazy loading)
        - store state collection on disk
        Args:
            agent_pool: Agents used to generate states
            num_players: number of agents in games that produce states collected here
            drop_actions: if True, only states will be returned/stored
            checkpoint_dir: if not None, states/actions will be written here
            target_agent: if provided, agent will participate in games for state production
            return_lazily: if True, data in __iter__ will be yielded
        """

        self.agent_pool = agent_pool
        self.num_players = num_players
        self.drop_actions = drop_actions
        self.checkpoint_dir = checkpoint_dir
        self.target_agent = target_agent
        self.return_laziliy = return_lazily


    def get_torch_IterableDataset(self):
        pass

    def collect(self):
        pass