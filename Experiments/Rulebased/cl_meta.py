import numpy as np
import os
import random
import sys
import getopt
import time
import rl_env
import numpy as np
import tensorflow as tf

import utils
from rulebased_agent import RulebasedAgent
from agents.random_agent import RandomAgent
from agents.simple_agent import SimpleAgent
from internal_agent import InternalAgent
from outer_agent import OuterAgent
from iggi_agent import IGGIAgent
from legal_random_agent import LegalRandomAgent
from flawed_agent import FlawedAgent
from piers_agent import PiersAgent
from van_den_bergh_agent import VanDenBerghAgent

AGENT_CLASSES = {'InternalAgent': InternalAgent,
                 'OuterAgent': OuterAgent, 'IGGIAgent': IGGIAgent, 'FlawedAgent': FlawedAgent,
                 'PiersAgent': PiersAgent, 'VanDenBerghAgent': VanDenBerghAgent}

class DataGenerator(object):
    """
    Data Generator capable of generating batches of Hanabi (state, action) pairs.
    A "class" is considered a rule based agent.
    """

    def __init__(self, num_classes, num_samples_per_class, num_meta_test_classes, num_meta_test_samples_per_class,
                 config={}):
        """
        Args:
          num_classes: Number of classes for classification (K-way)
          num_samples_per_class: num samples to generate per class in one batch
          num_meta_test_classes: Number of classes for classification (K-way) at meta-test time
          num_meta_test_samples_per_class: num samples to generate per class in one batch at meta-test time
          batch_size: size of meta batch size (e.g. number of functions)
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.num_meta_test_samples_per_class = num_meta_test_samples_per_class
        self.num_meta_test_classes = num_meta_test_classes

        # data_folder = config.get('data_folder', './omniglot_resized')
        # self.img_size = config.get('img_size', (28, 28))

        # self.dim_input = np.prod(self.img_size)
        # self.dim_output = self.num_classes

        character_folders = []

        random.seed(123)
        # random.shuffle(character_folders)
        num_val = 2
        num_train = 4
        # self.metatrain_character_folders = character_folders[: num_train]
        # self.metaval_character_folders = character_folders[
        #                                  num_train:num_train + num_val]
        # self.metatest_character_folders = character_folders[
        #                                   num_train + num_val:]
        self.meta_train_agents = AGENT_CLASSES[:num_train]


    def sample_batch(self, batch_type, batch_size, shuffle=True, swap=False):
        """
        Samples a batch for training, validation, or testing
        Args:
          batch_type: meta_train/meta_val/meta_test
          shuffle: randomly shuffle classes or not
          swap: swap number of classes (N) and number of samples per class (K) or not
        Returns:
          A a tuple of (1) Image batch and (2) Label batch where
          image batch has shape [B, N, K, 784] and label batch has shape [B, N, K, N] if swap is False
          where B is batch size, K is number of samples per class, N is number of classes
        """