# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple episode runner using the RL environment."""

from __future__ import print_function

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

# AGENT_CLASSES = {'SimpleAgent': SimpleAgent, 'RandomAgent': RandomAgent, 'InternalAgent': InternalAgent,
# 'OuterAgent': OuterAgent,'IGGIAgent':IGGIAgent,'LegalRandomAgent':LegalRandomAgent,'FlawedAgent':FlawedAgent,
# 'PiersAgent':PiersAgent, 'VanDenBerghAgent':VanDenBerghAgent}

AGENT_CLASSES = {'InternalAgent': InternalAgent,
                 'OuterAgent': OuterAgent, 'IGGIAgent': IGGIAgent, 'FlawedAgent': FlawedAgent,
                 'PiersAgent': PiersAgent, 'VanDenBerghAgent': VanDenBerghAgent}
AGENT_CLASSES_TEST = {'FlawedAgent': FlawedAgent, 'PiersAgent': PiersAgent}
LENIENT = False
N_AGENTS = 2


class Runner(object):
    """Runner class."""

    def __init__(self, flags, a1, a2, num_replay_samples=1e5):
        """Initialize runner."""
        self.flags = flags
        self.agent_config = {'players': flags['players']}
        self.environment = rl_env.make('Hanabi-Full', num_players=flags['players'])
        self.agent_class = AGENT_CLASSES[flags['agent_class']]  # unused
        self.class1 = AGENT_CLASSES[a1]
        self.class2 = AGENT_CLASSES[a2]
        self.a1 = a1
        self.a2 = a2
        self.num_replay_samples = num_replay_samples
        assert type(a1) == str
        self.replay_dictionary = dict()  # should store trajectories for each agent identifier
        self.replay_dictionary[a1] = {'states': list(), 'actions': list()}
        if a2 not in self.replay_dictionary.keys():
            self.replay_dictionary[a2] = {'states': list(), 'actions': list()}

    def run(self):
        """Run episodes."""
        rewards = []
        # agents = [self.agent_class(self.agent_config)
        #             for _ in range(self.flags['players'])]
        # agents = [a1,a2]
        i_move = 0

        while i_move < self.num_replay_samples:
            if np.random.uniform() <= 0.5:
                agents = [self.class1(self.agent_config), self.class2(self.agent_config)]
            else:
                agents = [self.class2(self.agent_config), self.class1(self.agent_config)]
            agent_id_to_str = dict()
            # make the agent names accessible via id while playing game
            for id, agent in enumerate(agents):
                if type(agent) == self.class1:
                    agent_id_to_str[id] = self.a1
                elif type(agent) == self.class2:
                    agent_id_to_str[id] = self.a2
                else:
                    raise KeyError
            observations = self.environment.reset()
            done = False
            episode_reward = 0
            first_turn = True
            while not done:
                for agent_id, agent in enumerate(agents):
                    observation = observations['player_observations'][agent_id]
                    if first_turn:
                        # print(first_turn)
                        # print(observation['current_player'])
                        first_turn = False
                    action = agent.act(observation)
                    if observation['current_player'] == agent_id:
                        assert action is not None
                        current_player_action = action
                        from_state = observation['vectorized']
                        # from_state = np.array(observation['vectorized'])
                        from_agent = agent_id_to_str[agent_id]

                        if not from_agent in self.replay_dictionary.keys():
                            raise KeyError
                        else:
                            self.replay_dictionary[from_agent]['states'].append(from_state)
                            # append one hot encoded action
                            move = self.environment._build_move(current_player_action)
                            int_action = self.environment.game.get_move_uid(move)  # 0 <= move_uid < max_moves()
                            one_hot_action = [0 for _ in range(self.environment.game.max_moves())]
                            one_hot_action[int_action] = 1
                            self.replay_dictionary[from_agent]['actions'].append(one_hot_action)
                            i_move += 1
                    else:
                        assert action is None
                # Make an environment step.
                # # print('Agent: {} action: {}'.format(observation['current_player'],
                #                                     current_player_action))
                observations, reward, done, unused_info = self.environment.step(
                    current_player_action)
                if (reward >= 0 or not LENIENT):
                    episode_reward += reward

            rewards.append(episode_reward)
            # print('Running episode: %d' % episode)
            # print('Reward of this episode: %d' % episode_reward)
            # print('Max Reward: %.3f' % max(rewards))
            # print('Average Reward: %.3f' % (sum(rewards)/(episode+1)))
        # for a in agents:
        #     a.rulebased.print_histogram()

        return rewards, self.replay_dictionary


def get_replay_dict_2_agents(agents):
    replay_dictionary = dict()
    for name in AGENT_CLASSES_TEST:
        replay_dictionary[name] = {'states': list(), 'actions': list()}

    # get trajectories
    for name1 in AGENT_CLASSES_TEST:
        for name2 in AGENT_CLASSES_TEST:
            runner = Runner(flags, name1, name2, num_replay_samples=1000)
            # todo check if integer values of actions are always in the same order
            rewards, trajectories = runner.run()
            replay_dictionary[name1]['states'].append(trajectories[name1]['states'])
            replay_dictionary[name1]['actions'].append(trajectories[name1]['actions'])
            if name1 != name2:
                replay_dictionary[name2]['states'].append(trajectories[name2]['states'])
                replay_dictionary[name2]['actions'].append(trajectories[name2]['actions'])
    return replay_dictionary


if __name__ == "__main__":
    flags = {'players': 2, 'num_episodes': 2, 'agent_class': 'OuterAgent'}
    options, arguments = getopt.getopt(sys.argv[1:], '',
                                       ['players=',
                                        'num_episodes=',
                                        'agent_class='])
    if arguments:
        sys.exit('usage: rl_env_example.py [options]\n'
                 '--players       number of players in the game.\n'
                 '--num_episodes  number of game episodes to run.\n'
                 '--agent_class   {}'.format(' or '.join(AGENT_CLASSES.keys())))
    for flag, value in options:
        flag = flag[2:]  # Strip leading --.
        flags[flag] = type(flags[flag])(value)
    results = []
    # results.append([name1, name2, reward])
    model = utils.load_default_model()

    replay_dictionary = get_replay_dict_2_agents(AGENT_CLASSES_TEST)
    X, Y = utils.train_data_from_replay_dict(replay_dictionary)

    for train_epoch in range(5):
        # TRAIN (on former test set)
        model.fit(X, Y, epochs=20)

        replay_dictionary = get_replay_dict_2_agents(AGENT_CLASSES_TEST)
        # EVAL (with new test set)
        X, Y = utils.train_data_from_replay_dict(replay_dictionary)
        model.evaluate(X, Y)