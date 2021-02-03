import traceback

COLORS_INV = ['B', 'W', 'G', 'Y', 'R']
RANKS_INV = [4, 3, 2, 1, 0]
C_COUNT = [3, 2, 2, 2, 1]


def stringify_env_config(cfg):
  c = str(cfg['colors'])
  r = str(cfg['ranks'])
  p = str(cfg['players'])
  h = str(cfg['hand_size'])
  i = str(cfg['max_information_tokens'])
  l = str(cfg['max_life_tokens'])
  o = str(cfg['observation_type'])
  return f'p{p}_h{h}_c{c}_r{r}_i{i}_l{l}_o{o}'





def get_observation_length(cfg):
  num_colors = cfg['colors']
  num_ranks = cfg['ranks']
  num_players = cfg['players']
  max_deck_size = 0
  for r in range(num_ranks):
    max_deck_size += C_COUNT[r] * num_colors
  max_info_tokens = cfg['max_information_tokens']
  max_life_tokens = cfg['max_life_tokens']
  hand_size = 4 if num_players in [4, 5] else 5

  bits_per_card = num_colors * num_ranks

  hands_bit_length = (num_players - 1) * hand_size * bits_per_card + num_players

  board_bit_length = max_deck_size - num_players * \
                     hand_size + num_colors * num_ranks \
                     + max_info_tokens + max_life_tokens

  discard_pile_bit_length = max_deck_size

  last_action_bit_length = num_players + 4 + num_players + \
                           num_colors + num_ranks \
                           + hand_size + hand_size + bits_per_card + 2

  card_knowledge_bit_length = num_players * hand_size * \
                              (bits_per_card + num_colors + num_ranks)

  return hands_bit_length + board_bit_length + discard_pile_bit_length \
         + last_action_bit_length + card_knowledge_bit_length


def get_max_actions(cfg):
  hand_size = 4 if cfg['players'] in [4, 5] else 5
  return 2 * hand_size + (cfg['colors'] + cfg['ranks']) * cfg['players']



def to_int(cfg, action_dict):
  try:
    action_type = action_dict['action_type']
  except Exception:
    traceback.print_exc()
    print(f'got action = {action_dict}')
    exit(1)
  if action_type == 'DISCARD':
    return action_dict['card_index']
  elif action_type == 'PLAY':
    return cfg['hand_size'] + action_dict['card_index']
  elif action_type == 'REVEAL_COLOR':
    color_offset = (2 * cfg['hand_size'])
    return color_offset + action_dict['target_offset'] * cfg['colors'] - (COLORS_INV.index(action_dict['color'])) - 1
  elif action_type == 'REVEAL_RANK':
    rank_offset = 2 * cfg['hand_size'] + (cfg['players'] - 1) * cfg['colors']
    return rank_offset + action_dict['target_offset'] * cfg['ranks'] - (RANKS_INV[action_dict['rank']]) - 1
  else:
    raise ValueError(f'action_dict was {action_dict}')


# todo 1: visualize pairwise training and test accuracies
# todo 2: increase the number of different agent classes for classification
# todo 3: visualize the test accuracy as a function of the number of agents
# todo 4: apply Relevance Propagation methods, to find out which are informative features
# todo 5: visualize these heatmaps using horizontal encoding
# todo 6: apply Meta Learning techniques
# todo 7: hand in thesis
