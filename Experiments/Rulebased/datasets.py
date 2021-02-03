import torch
import sqlite3
import enum
import pickle
import traceback


class PoolOfStates(torch.utils.data.Dataset):
  def __init__(self,
               from_db_path='./database_test.db',
               load_state_as_type='dict',  # or 'dict'
               drop_actions=False,
               # size=int(1e5),
               target_table='pool_of_state_dicts'
               ):
    torch.utils.data.Dataset.__init__(self)
    self._from_db_path = from_db_path  # path to .db file
    self._drop_actions = drop_actions
    # self._size = size
    self._target_table = target_table
    self._connection = sqlite3.connect(self._from_db_path)
    assert load_state_as_type in ['torch.FloatTensor', 'dict'], 'states must be either torch.FloatTensor or dict'
    self._load_state_as_type = load_state_as_type
    self.QUERY_VARS = ['num_players',
                       'agent',
                       'current_player',
                       'current_player_offset',
                       'deck_size',
                       'discard_pile',
                       'fireworks',
                       'information_tokens',
                       'legal_moves',
                       'life_tokens',
                       'observed_hands',
                       'card_knowledge',
                       'vectorized',
                       'pyhanabi']
    actions = [] if drop_actions else ['int_action', 'dict_action ']
    self.QUERY_VARS += actions

  class QueryCols(enum.IntEnum):
    num_players = 0
    agent = 1
    current_player = 2
    current_player_offset = 3
    deck_size = 4
    discard_pile = 5
    fireworks = 6
    information_tokens = 7
    legal_moves = 8
    life_tokens = 9
    observed_hands = 10
    card_knowledge = 11
    vectorized = 12
    pyhanabi = 13
    int_action = 14
    dict_action = 15

  def _build_query(self, n_rows, table='pool_of_state_dicts', seed='', pick_at_random=False) -> str:
    try:
      query_cols = [col + ', ' for col in self.QUERY_VARS]
      query_cols[-1] = query_cols[-1][:-2]  # remove last ', '
      """ Unfortunately sqlite3 doesnt support seeding the RANDOM() call """
      random = 'ORDER BY RANDOM()' if pick_at_random else ''
      query_string = ['SELECT '] + query_cols + [' from ' + table] + [
        f' WHERE _ROWID_ IN (SELECT _ROWID_ FROM {table} {random}'] + [f' LIMIT {n_rows})']
      return "".join(query_string)
    except Exception as e:
      print(e)
      print(traceback.print_exc())
      exit(1)

  def get_eagerly(self, n_rows, batch_size=1, pyhanabi_as_bytes=True, pick_at_random=False, random_seed=None):
    dataset = []
    cursor = self._connection.cursor()
    # query_string = self._build_query(str(random_seed) if random_seed else '')
    query_string = self._build_query(n_rows=n_rows,
                                     pick_at_random=pick_at_random,
                                     seed=str(random_seed) if random_seed else '')
    # query database with all the information necessary to build the observation_dictionary
    cursor.execute(query_string)
    # parse query
    it = 0
    batch_it = 0
    batch = []
    for row in cursor:  # database row
      # build observation_dict from row
      if batch_it < batch_size:
        batch.append(self._parse_row_to_dict(row, pyhanabi_as_bytes=pyhanabi_as_bytes))
        batch_it += 1
      else:
        dataset.append(batch)
        batch = [self._parse_row_to_dict(row, pyhanabi_as_bytes=pyhanabi_as_bytes)]
        batch_it = 1
      it += 1
      if it % 1000 == 0:
        # print(f'loaded {it} rows')
        pass
      if it >= n_rows:
        break
    return dataset

  def _parse_row_to_dict(self, row, pyhanabi_as_bytes=False):
    obs_dict = {}
    # assign columns of query to corresponding key in observation_dict
    obs_dict[self.QueryCols.num_players.name] = row[self.QueryCols.num_players.value]
    obs_dict[self.QueryCols.agent.name] = row[self.QueryCols.agent.value]
    obs_dict[self.QueryCols.current_player.name] = row[self.QueryCols.current_player.value]
    obs_dict[self.QueryCols.current_player_offset.name] = row[self.QueryCols.current_player_offset.value]
    obs_dict[self.QueryCols.deck_size.name] = row[self.QueryCols.deck_size.value]
    obs_dict[self.QueryCols.discard_pile.name] = eval(row[self.QueryCols.discard_pile.value])
    obs_dict[self.QueryCols.fireworks.name] = eval(row[self.QueryCols.fireworks.value])
    obs_dict[self.QueryCols.information_tokens.name] = row[self.QueryCols.information_tokens.value]
    obs_dict[self.QueryCols.legal_moves.name] = eval(row[self.QueryCols.legal_moves.value])
    obs_dict[self.QueryCols.life_tokens.name] = row[self.QueryCols.life_tokens.value]
    obs_dict[self.QueryCols.observed_hands.name] = eval(row[self.QueryCols.observed_hands.value])
    obs_dict[self.QueryCols.card_knowledge.name] = eval(row[self.QueryCols.card_knowledge.value])
    obs_dict[self.QueryCols.vectorized.name] = eval(row[self.QueryCols.vectorized.value])
    if pyhanabi_as_bytes:
      obs_dict[self.QueryCols.pyhanabi.name] = row[self.QueryCols.pyhanabi.value]
    else:
      obs_dict[self.QueryCols.pyhanabi.name] = pickle.loads(row[self.QueryCols.pyhanabi.value])

    if not self._drop_actions:
      obs_dict[self.QueryCols.int_action.name] = row[self.QueryCols.int_action.value]
      obs_dict[self.QueryCols.dict_action.name] = row[self.QueryCols.dict_action.value]

    return obs_dict

#
# class IterablePoolOfStatesFromDatabase(torch.utils.data.IterableDataset, PoolOfStatesFromDatabase):
#   def __init__(self, from_db_path='./database_test.db',
#                load_state_as_type='dict',  # or 'dict'
#                drop_actions=False,
#                size=int(1e5),
#                target_table='pool_of_state_dicts',
#                batch_size=1):
#     torch.utils.data.IterableDataset.__init__(self)
#     PoolOfStatesFromDatabase.__init__(self, from_db_path=from_db_path,
#                                       load_state_as_type=load_state_as_type,  # or 'dict'
#                                       drop_actions=drop_actions,
#                                       size=size,
#                                       target_table=target_table
#                                       )
#     self.batch_size = batch_size
#
#   def _yield_dict(self):
#     cursor = self._connection.cursor()
#     query_string = self._build_query()
#     # query database with all the information necessary to build the observation_dictionary
#     cursor.execute(query_string)
#     # parse query
#     for row in cursor:  # database row
#       # build observation_dict from row
#       obs_dict = self._parse_row_to_dict(row)
#       # yield row by row the observation_dictionary unpacked from that row
#       yield obs_dict
#
#   def get_rows_lazily(self):
#     if self._load_state_as_type == 'dict':
#       return self._yield_dict()
#     elif self._load_state_as_type == 'torch.FloatTensor':
#       raise NotImplementedError
#     else:
#       raise NotImplementedError
#
#   @staticmethod
#   def _create_batch(from_list):
#     return zip(*from_list)
#
#   def __iter__(self):
#     # if self._load_lazily:
#     #   return iter(self._create_batch([self.get_rows_lazily() for _ in range(self._batch_size)]))
#     # else:
#     #   # todo here, we could load eagerly to distribute a large dataset via ray.tune.run_with_parameters()
#     #   # but I think the lazy implementation is parallelized quite nicely, because of the yield
#     #   raise NotImplementedError
#     return iter(self._create_batch([self.get_rows_lazily() for _ in range(self.batch_size)]))
#
#
# class MapStylePoolOfStatesFromDatabase(torch.utils.data.Dataset, PoolOfStatesFromDatabase):
#   # query all rows
#   # parse row to dict
#   # create list and return iterator over that list
#   def __init__(self, from_db_path='./database_test.db',
#                load_state_as_type='dict',  # or 'dict'
#                drop_actions=False,
#                size=int(1e5),
#                target_table='pool_of_state_dicts',
#                batch_size=1):
#     torch.utils.data.Dataset.__init__(self)
#     PoolOfStatesFromDatabase.__init__(self, from_db_path=from_db_path,
#                                       load_state_as_type=load_state_as_type,  # or 'dict'
#                                       drop_actions=drop_actions,
#                                       size=size,
#                                       target_table=target_table
#                                       )