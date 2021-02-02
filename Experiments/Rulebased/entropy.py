import pickle
import time
import rulebased_agent as ra
from Experiments.Rulebased.cl2 import StateActionCollector, AGENT_CLASSES
from typing import NamedTuple
import sqlite3
from sklearn.metrics import normalized_mutual_info_score
import traceback
import numpy as np


import matplotlib.pyplot as plt
class Agent(NamedTuple):
  name: str
  instance: ra.RulebasedAgent


def get_states(num_states):
  start = time.time()
  statecollector = StateActionCollector(AGENT_CLASSES, 3)
  states = statecollector.collect(num_states_to_collect=num_states,
           keep_obs_dict=True,
           keep_agent=False)
  print(f'gathering took {time.time() - start} seconds')

  return [pickle.loads(s) for s in states]


def get_actions(agent, pool_of_states):
  hand_size = 5
  num_players = 3
  num_colors = 5
  num_ranks = 5
  COLORS = ['R', 'Y', 'G', 'W', 'B']
  COLORS_INV = ['B', 'W', 'G', 'Y', 'R']
  RANKS_INV = [4, 3, 2, 1, 0]
  color_offset = (2 * hand_size)
  rank_offset = color_offset + (num_players - 1) * num_colors

  def to_int(action_dict):
    action_type = action_dict['action_type']
    if action_type == 'DISCARD':
      return action_dict['card_index']
    elif action_type == 'PLAY':
      return hand_size + action_dict['card_index']
    elif action_type == 'REVEAL_COLOR':
      return color_offset + action_dict['target_offset'] * num_colors - (COLORS_INV.index(action_dict['color'])) - 1
    elif action_type == 'REVEAL_RANK':
      return rank_offset + action_dict['target_offset'] * num_ranks - (RANKS_INV[action_dict['rank']]) - 1
    else:
      raise ValueError(f'action_dict was {action_dict}')

  actions = []

  for state in pool_of_states:
    actions.append(to_int(agent.act(state)))
  return actions


def mutual_information(agent_1, agent_2, pool_of_states):
  """ pool_of_states may be passed eagerly as an array or as a database cursor """
  if isinstance(pool_of_states, sqlite3.Cursor):
    # load states from database
    raise NotImplementedError

  actions_1 = get_actions(agent_1, pool_of_states)
  actions_2 = get_actions(agent_2, pool_of_states)
  # print(f'for agents {agent_1} and {agent_2} we have '
  #       f'\n actions_1 = {actions_1} and '
  #       f'\nactions_2 = {actions_2}')
  return normalized_mutual_info_score(actions_1, actions_2)


def main():
  pool_of_states = get_states(num_states=1000)
  agent_1 = InternalAgent({'players': 3})
  agent_2 = OuterAgent({'players': 3})
  agents = list(AGENT_CLASSES.values())
  num_agents = len(agents)
  mutual_informations = np.zeros((num_agents, num_agents))
  for i in range(num_agents):
    for j in range(num_agents):
      try:
        mutual_informations[i][j] = mutual_information(agents[i]({'players': 3}), agents[j]({'players': 3}), pool_of_states)
        print(f'finished computation for i,j = {i,j}')
      except Exception as e:
        print(e)
        print(traceback.print_exc())

  print(f'matrix = {mutual_informations}')


if __name__ == '__main__':
  main()
  # matrix_1000_old = [[0.77840222, 0.65183344, 0.3942758, 0.11615003, 0.34446092, 0.23851854],
  #           [0.65118969, 0.98014517, 0.5115872, 0.11186995, 0.43544002, 0.34943244],
  #           [0.40239239, 0.50333585, 0.7917706, 0.10762333, 0.65209376, 0.46792381],
  #           [0.12877865, 0.11268436, 0.10734912, 0.93157149, 0.19721827, 0.16013667],
  #           [0.34616255, 0.43560051, 0.65755464, 0.19553393, 0.85717321, 0.59291364],
  #           [0.23793211, 0.35260278, 0.46788897, 0.15881293, 0.59135198, 0.92188265]]
 #  matrix_5000_old = [[0.76501058, 0.61816794, 0.36537244, 0.07471123, 0.31679159, 0.18604767],
 # [0.61559739, 0.9824943,  0.48863918, 0.06765159, 0.45820619, 0.31249469],
 # [0.36102607, 0.4883036,  0.72680195, 0.05253612, 0.6529997,  0.41711241],
 # [0.07288302, 0.06549227, 0.05344163, 0.89219117, 0.11773575, 0.08305062],
 # [0.31735259, 0.4553607,  0.65125179, 0.11771185, 0.83070411, 0.5125517 ],
 # [0.18667698, 0.31076851, 0.41327228, 0.08260657, 0.51259434, 0.88882435]]

 #  plt.xticks(ticks=np.arange(len(AGENT_CLASSES.keys())), labels=list(AGENT_CLASSES.keys()), rotation=20)
 #  plt.yticks(ticks=np.arange(len(AGENT_CLASSES.keys())), labels=list(AGENT_CLASSES.keys()))
 #  plt.colorbar(plt.imshow(matrix_5000))
 #  plt.show()
