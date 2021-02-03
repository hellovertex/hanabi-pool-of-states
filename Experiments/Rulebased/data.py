import os
from cl2 import StateActionCollector, AGENT_CLASSES
from typing import Dict, Optional
import rulebased_agent as ra


class StateActionWriter:
  """ - Collects states and actions using the StateActionCollector
      - writes them to database
  """

  def __init__(self,
               agent_classes: Dict[str, ra.RulebasedAgent],
               hanabi_game_config: Dict,
               num_players: int,
               # max_size_per_iter: int = 100,
               target_agent: Optional[str] = None,
               ):
    self._data_collector = StateActionCollector(hanabi_game_config=hanabi_game_config,
                                                agent_classes=agent_classes,
                                                target_agent=target_agent)

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


def maybe_create_and_populate_database(db_path, env_config, using_n_states):
  if not os.path.exists(db_path):
    print(f'database created (now)')
    print(f'collecting states... This may take a long time')
    writer = StateActionWriter(agent_classes=AGENT_CLASSES,
                               hanabi_game_config=env_config,
                               num_players=env_config['players'])
    writer.collect_and_write_to_database(db_path, using_n_states)
    print(f'Done!')
  else:
    print('Database existed already, skipping state collection...')
