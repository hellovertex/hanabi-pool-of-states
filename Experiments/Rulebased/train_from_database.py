from functools import partial
import torch
import sqlite3
import numpy as np
from torch.utils.data import DataLoader
import os
from time import time
import pickle
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune import CLIReporter
from torch.utils.tensorboard import SummaryWriter
import enum
import torch.optim as optim
# project lvl imports
import rulebased_agent as ra
from Experiments.Rulebased.data import StateActionWriter
from Experiments.Rulebased.flawed_agent import FlawedAgent
from cl2 import StateActionCollector, AGENT_CLASSES, to_int
import traceback
import model

DEBUG = True
USE_RAY = False
if DEBUG:
  LOG_INTERVAL = 10
  EVAL_INTERVAL = 20
  NUM_EVAL_STATES = 500
  # for model selection
  GRACE_PERIOD = 1
  MAX_T = 100
  MAX_TRAIN_ITERS = 1000
  NUM_SAMPLES = 1
  BATCH_SIZE = 4
else:
  LOG_INTERVAL = 100
  EVAL_INTERVAL = 500
  NUM_EVAL_STATES = 300
  GRACE_PERIOD = int(5e4)  # tune.report gets only called at EVAL_INTERVAL anyway
  MAX_T = int(150e3)  # tune.report gets only called at EVAL_INTERVAL anyway
  NUM_SAMPLES = 10
  MAX_TRAIN_ITERS = 100000
  BATCH_SIZE = 16  # tune.grid_search([8, 1])

KEEP_CHECKPOINTS_NUM = 50
VERBOSE = 1
from_db_path_notebook = '/home/cawa/Documents/github.com/hellovertex/hanabi-ad-hoc-learning/Experiments/Rulebased/database_test.db'
from_db_path_desktop = '/home/hellovertex/Documents/github.com/hellovertex/hanabi-ad-hoc-learning/Experiments/Rulebased/database_test.db'
FROM_DB_PATH = from_db_path_desktop


def stringify_env_config(cfg):
  c = str(cfg['colors'])
  r = str(cfg['ranks'])
  p = str(cfg['players'])
  h = str(cfg['hand_size'])
  i = str(cfg['max_information_tokens'])
  l = str(cfg['max_life_tokens'])
  o = str(cfg['observation_type'])
  return f'p{p}_h{h}_c{c}_r{r}_i{i}_l{l}_o{o}'


class PoolOfStatesFromDatabase:
  def __init__(self, from_db_path='./database_test.db',
               load_state_as_type='dict',  # or 'dict'
               drop_actions=False,
               size=int(1e5),
               target_table='pool_of_state_dicts'
               ):
    self._from_db_path = from_db_path  # path to .db file
    self._drop_actions = drop_actions
    self._size = size
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

  def _build_query(self, table='pool_of_state_dicts', seed='', pick_at_random=False) -> str:
    try:
      query_cols = [col + ', ' for col in self.QUERY_VARS]
      query_cols[-1] = query_cols[-1][:-2]  # remove last ', '
      """ Unfortunately sqlite3 doesnt support seeding the RANDOM() call """
      random = 'ORDER BY RANDOM()' if pick_at_random else ''
      query_string = ['SELECT '] + query_cols + [' from ' + table] + [
        f' WHERE _ROWID_ IN (SELECT _ROWID_ FROM {table} {random}'] + [f' LIMIT {self._size})']
      return "".join(query_string)
    except Exception as e:
      print(e)
      print(traceback.print_exc())
      exit(1)

  def get_eagerly(self, batch_length=1, pyhanabi_as_bytes=True, pick_at_random=False, random_seed=None):
    dataset = []
    cursor = self._connection.cursor()
    # query_string = self._build_query(str(random_seed) if random_seed else '')
    query_string = self._build_query(pick_at_random=pick_at_random, seed=str(random_seed) if random_seed else '')
    # query database with all the information necessary to build the observation_dictionary
    cursor.execute(query_string)
    # parse query
    it = 0
    batch_it = 0
    batch = []
    for row in cursor:  # database row
      # build observation_dict from row
      if batch_it < batch_length:
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
      if it >= self._size:
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


class IterablePoolOfStatesFromDatabase(torch.utils.data.IterableDataset, PoolOfStatesFromDatabase):
  def __init__(self, from_db_path='./database_test.db',
               load_state_as_type='dict',  # or 'dict'
               drop_actions=False,
               size=int(1e5),
               target_table='pool_of_state_dicts',
               batch_size=1):
    torch.utils.data.IterableDataset.__init__(self)
    PoolOfStatesFromDatabase.__init__(self, from_db_path=from_db_path,
                                      load_state_as_type=load_state_as_type,  # or 'dict'
                                      drop_actions=drop_actions,
                                      size=size,
                                      target_table=target_table
                                      )
    self.batch_size = batch_size

  def _yield_dict(self):
    cursor = self._connection.cursor()
    query_string = self._build_query()
    # query database with all the information necessary to build the observation_dictionary
    cursor.execute(query_string)
    # parse query
    for row in cursor:  # database row
      # build observation_dict from row
      obs_dict = self._parse_row_to_dict(row)
      # yield row by row the observation_dictionary unpacked from that row
      yield obs_dict

  def get_rows_lazily(self):
    if self._load_state_as_type == 'dict':
      return self._yield_dict()
    elif self._load_state_as_type == 'torch.FloatTensor':
      raise NotImplementedError
    else:
      raise NotImplementedError

  @staticmethod
  def _create_batch(from_list):
    return zip(*from_list)

  def __iter__(self):
    # if self._load_lazily:
    #   return iter(self._create_batch([self.get_rows_lazily() for _ in range(self._batch_size)]))
    # else:
    #   # todo here, we could load eagerly to distribute a large dataset via ray.tune.run_with_parameters()
    #   # but I think the lazy implementation is parallelized quite nicely, because of the yield
    #   raise NotImplementedError
    return iter(self._create_batch([self.get_rows_lazily() for _ in range(self.batch_size)]))


class MapStylePoolOfStatesFromDatabase(torch.utils.data.Dataset, PoolOfStatesFromDatabase):
  # query all rows
  # parse row to dict
  # create list and return iterator over that list
  def __init__(self, from_db_path='./database_test.db',
               load_state_as_type='dict',  # or 'dict'
               drop_actions=False,
               size=int(1e5),
               target_table='pool_of_state_dicts',
               batch_size=1):
    torch.utils.data.Dataset.__init__(self)
    PoolOfStatesFromDatabase.__init__(self, from_db_path=from_db_path,
                                      load_state_as_type=load_state_as_type,  # or 'dict'
                                      drop_actions=drop_actions,
                                      size=size,
                                      target_table=target_table
                                      )


def eval_fn(env_config, net, eval_loader, criterion, target_agent, num_states):
  # load pickled observations and get vectorized and compute action and eval with that
  start = time()
  observations_pickled = eval_loader.collect(num_states_to_collect=num_states,
                                             keep_obs_dict=True,
                                             keep_agent=False)
  assert len(
    observations_pickled) == num_states, f'len(observations_pickled)={len(observations_pickled)} and num_states = {num_states}'
  # print(f'Collecting eval states took {time() - start} seconds')
  correct = 0
  running_loss = 0
  with torch.no_grad():
    for obs in observations_pickled:
      observation = pickle.loads(obs)
      action = torch.LongTensor([to_int(env_config, target_agent.act(observation))])
      prediction = net(torch.FloatTensor(observation['vectorized'])).reshape(1, -1)
      # loss
      running_loss += criterion(prediction, action)
      # accuracy
      correct += torch.sum(torch.max(prediction, 1)[1] == action)

    return 100 * running_loss / num_states, 100 * correct.item() / num_states


def train_eval(config,
               env_config,
               using_num_states=int(2e3),
               conn=None,
               checkpoint_dir=None,
               from_db_path=None,
               target_table='pool_of_state_dicts',
               log_interval=100,
               eval_interval=1000,
               num_eval_states=100,
               max_train_steps=np.inf,
               use_ray=True,
               dataset=None,
               pyhanabi_as_bytes=False):
  target_agent_cls = config['agent']
  lr = config['lr']
  num_hidden_layers = config['num_hidden_layers']
  layer_size = config['layer_size']
  batch_size = config['batch_size']
  num_players = config['num_players']
  target_agent = target_agent_cls(config['agent_config'])

  if from_db_path is None:
    raise NotImplementedError("Todo: Implement the database setup before training on new machines. ")

  def _train_loader():
    """ Returns first n rows of database """
    return MapStylePoolOfStatesFromDatabase(from_db_path=from_db_path, size=using_num_states).get_eagerly(
      pyhanabi_as_bytes=True, batch_length=1, random_seed=42)

  trainloader = _train_loader()
  testloader = StateActionCollector(env_config, AGENT_CLASSES, num_players)

  net = model.get_model(observation_size=956,  # todo derive this from game_config
                        num_actions=30,
                        num_hidden_layers=num_hidden_layers,
                        layer_size=layer_size)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=lr)
  it = 0
  if checkpoint_dir:
    print("Loading from checkpoints")
    path = os.path.join(checkpoint_dir, "checkpoint")
    model_state, optimizer_state = torch.load(path)
    net.load_state_dict(model_state())
    optimizer.load_state_dict(optimizer_state())
  epoch = 1
  moving_acc = 0
  eval_it = 0
  if not use_ray:
    ckptdir = checkpoint_dir if checkpoint_dir else ''
    writer = SummaryWriter(log_dir=ckptdir + 'manual_train/')

  def _parse_batch(b):
    ret = []
    for obs in b:
      d = dict(obs)
      if pyhanabi_as_bytes:
        d['pyhanabi'] = pickle.loads(obs['pyhanabi'])
      ret.append(d)
    return ret

  while True:
    try:
      for batch_raw in trainloader:
        batch = _parse_batch(batch_raw)
        actions = torch.LongTensor([to_int(env_config, target_agent.act(obs)) for obs in batch])
        vectorized = torch.FloatTensor([obs['vectorized'] for obs in batch])
        optimizer.zero_grad()
        outputs = net(vectorized).reshape(batch_size, -1)
        loss = criterion(outputs, actions)

        loss.backward()
        optimizer.step()

        if it % log_interval == 0 and not use_ray:
          print(f'Iteration {it}...')
        if it % eval_interval == 0:
          loss, acc = eval_fn(env_config, net=net, eval_loader=testloader, criterion=criterion, target_agent=target_agent,
                              num_states=num_eval_states)
          moving_acc += acc
          eval_it += 1
          if not use_ray:
            # print(f'Loss at iteration {it} is {loss}, and accuracy is {moving_acc / eval_it} %')
            print(f'Loss at iteration {it} is {loss}, and accuracy is {acc} %')
            writer.add_scalar('Loss/test', loss, it)
            writer.add_scalar('Accuracy/test', acc, it)
            if checkpoint_dir:
              print(f'saving model to {checkpoint_dir}')
              path = os.path.join(checkpoint_dir, f'checkpoint_+{it}')
              torch.save((net.state_dict, optimizer.state_dict), path)
          else:
            # tune.report(training_iteration=it, loss=loss, acc=moving_acc / eval_it)
            tune.report(training_iteration=it, loss=loss, acc=acc)
            # checkpoint frequency may be handled by ray if we remove checkpointing here
            with tune.checkpoint_dir(step=it) as checkpoint_dir:
              path = os.path.join(checkpoint_dir, 'checkpoint')
              torch.save((net.state_dict, optimizer.state_dict), path)
        it += 1
        if it > max_train_steps:
          return
      trainloader = _train_loader()
    except Exception as e:
      if isinstance(e, StopIteration):
        # database has been read fully, start over
        # trainloader = dataset
        trainloader = _train_loader()
        epoch += 1
        continue
      else:
        print(e)
        print(traceback.print_exc())
        raise e


def run_train_eval_with_ray(env_config, db_path, name, scheduler, search_space, metric, mode, log_interval, eval_interval,
                            num_eval_states,
                            num_samples, max_train_steps):
  train_fn = tune.with_parameters(train_eval,
                                  from_db_path=db_path,
                                  env_config=env_config,
                                  target_table='pool_of_state_dicts',
                                  log_interval=log_interval,
                                  eval_interval=eval_interval,
                                  num_eval_states=num_eval_states,
                                  max_train_steps=max_train_steps,
                                  use_ray=True,
                                  pyhanabi_as_bytes=True)

  analysis = tune.run(train_fn,
                      metric=metric,
                      mode=mode,
                      config=search_space,
                      name=name,
                      num_samples=num_samples,
                      keep_checkpoints_num=KEEP_CHECKPOINTS_NUM,
                      verbose=VERBOSE,
                      trial_dirname_creator=None,
                      trial_name_creator=None,
                      # stopipp=ray.tune.EarlyStopping(metric='acc', top=5, patience=1, mode='max'),
                      scheduler=scheduler,
                      progress_reporter=CLIReporter(metric_columns=["loss", "acc", "training_iteration"]),
                      # trial_dirname_creator=trial_dirname_creator_fn
                      )
  best_trial = analysis.get_best_trial("acc", "max")
  print(best_trial.config)
  print(best_trial.checkpoint)

  # print(analysis.best_dataframe['acc'])
  return analysis


def select_best_model(env_config,
                      db_path,
                      name,
                      agentcls,
                      metric,
                      mode,
                      grace_period,
                      max_t,
                      num_samples,
                      lr,
                      layer_size,
                      batch_size):
  scheduler = ASHAScheduler(time_attr='training_iteration',
                            # metric=metric,
                            grace_period=grace_period,
                            # mode=mode,
                            max_t=max_t)  # current implementation raises stop iteration when db is finished
  # todo if necessary, build the search space from call params
  search_space = {'agent': agentcls,  # tune.grid_search(list(AGENT_CLASSES.values())),  #  agentcls,
                  'lr': lr,  # learning rate seems to be best in [2e-3, 4e-3], old [1e-4, 1e-1]
                  'num_hidden_layers': 1,  # tune.grid_search([1, 2]),
                  # 'layer_size': tune.grid_search([64, 96, 128, 196, 256, 376, 448, 512]),
                  # 'layer_size': tune.grid_search([64, 96, 128, 196, 256]),
                  # 'layer_size': tune.grid_search([64, 128, 256]),
                  'layer_size': layer_size,
                  'batch_size': batch_size,  # tune.choice([4, 8, 16, 32]),
                  'num_players': env_config['players'],
                  'agent_config': {'players': env_config['players']}
                  }
  return run_train_eval_with_ray(env_config=env_config,
                                 db_path=db_path,
                                 name=name,
                                 scheduler=scheduler,
                                 search_space=search_space,
                                 metric=metric,  # handled by scheduler, mutually exclusive
                                 mode=mode,  # handled by scheduler, mutually exclusive
                                 log_interval=LOG_INTERVAL,
                                 eval_interval=EVAL_INTERVAL,
                                 num_eval_states=NUM_EVAL_STATES,
                                 num_samples=num_samples,
                                 max_train_steps=max(max_t + 1, MAX_TRAIN_ITERS)
                                 )


def tune_best_model(db_path, env_config, experiment_name, analysis, with_pbt, max_train_steps=1e6):
  # todo: careful there are still bugs here
  # load config from analysis_obj
  best_trial = analysis.get_best_trial("acc", "max")
  config = best_trial.config
  print(analysis.best_checkpoint)
  print(analysis.best_logdir)
  print(analysis.best_result)
  print(analysis.best_result_df)
  print(config)
  # create search space for pbt
  search_space = {
    "lr": tune.uniform(1e-2, 1e-4),
  }
  # create pbt scheduler, see https://docs.ray.io/en/master/tune/api_docs/schedulers.html
  pbt = PopulationBasedTraining(
    time_attr="training_iteration",
    # metric="acc",
    # mode="max",
    perturbation_interval=500,  # every 10 `time_attr` units
    # (training_iterations in this case)
    hyperparam_mutations=search_space)
  # run pbt with final checkpoint dir
  return run_train_eval_with_ray(name=experiment_name,
                                 env_config=env_config,
                                 db_path=db_path,
                                 scheduler=pbt,
                                 search_space=config,
                                 metric='acc',  # handled by scheduler, mutually exclusive
                                 mode='max',  # handled by scheduler, mutually exclusive
                                 log_interval=LOG_INTERVAL,
                                 eval_interval=EVAL_INTERVAL,
                                 num_eval_states=NUM_EVAL_STATES,
                                 num_samples=NUM_SAMPLES,
                                 max_train_steps=max_train_steps
                                 )


def train_best_model(name, analysis, train_steps, from_db_path):
  # todo load train_eval fn with checkpoint_dir=best_trial.checkpoint_dir and continue training
  best_trial = analysis.get_best_trial('acc', 'max')
  checkpoint_dir = best_trial.checkpoint.value
  print(f'starting trainign in {checkpoint_dir}')
  config = best_trial.config
  # train_eval(config=config, checkpoint_dir=checkpoint_dir,use_ray=False, max_train_steps=train_steps, from_db_path=from_db_path, num_eval_states=NUM_EVAL_STATES)
  train_fn = partial(train_eval, max_train_steps=train_steps, from_db_path=from_db_path,
                     num_eval_states=NUM_EVAL_STATES)

  def _trial_dirname_creator(trial):
    return f'Manual_training_lr={trial.config["lr"]}' + f'_layer_size={trial.config["layer_size"]}'

  def _trial_name_creator(trial):
    return 'only_trial_' + str(trial.config['agent'])

  tune.run(train_fn,
           metric='acc',
           mode='max',
           config=config,
           name=name,
           num_samples=1,
           keep_checkpoints_num=1,
           verbose=VERBOSE,
           trial_dirname_creator=_trial_dirname_creator,
           trial_name_creator=None,
           # stopipp=ray.tune.EarlyStopping(metric='acc', top=5, patience=1, mode='max'),
           scheduler=None,
           progress_reporter=CLIReporter(metric_columns=["loss", "acc", "training_iteration"]),
           # trial_dirname_creator=trial_dirname_creator_fn
           )


def main(hanabi_config):
  # todo check if database exists for corresponding config, otherwise create it using 500k states
  db_path = f'./database_{stringify_env_config(hanabi_config)}.db'
  if not os.path.exists(db_path):
    # assuming database is not setup already, so create it and collect 500k states [takes a long time]
    writer = StateActionWriter(agent_classes=AGENT_CLASSES,
                               hanabi_game_config=hanabi_config,
                               num_players=3)
    writer.collect_and_write_to_database(db_path, int(5e3))
  #
  # for agentname, agentcls in AGENT_CLASSES.items():
  #   if USE_RAY:
  #     # dataset =
  #     best_model_analysis = select_best_model(env_config=hanabi_config,
  #                                             db_path=db_path,
  #                                             name=agentname,
  #                                             agentcls=agentcls,
  #                                             metric='acc',
  #                                             mode='max',
  #                                             grace_period=GRACE_PERIOD,
  #                                             max_t=MAX_T,
  #                                             num_samples=NUM_SAMPLES,
  #                                             lr=tune.loguniform(1e-4, 5e-3),
  #                                             layer_size=tune.grid_search([64, 128, 256]),
  #                                             batch_size=BATCH_SIZE)  # USE DEFAULTS for metric etc
  #     # todo maybe create two tune.Experiment instances for these
  #     print(f'Result written to {best_model_analysis.get_best_trial("acc", "max").checkpoint.value}')
  #     # train_best_model(name=agentname, analysis=best_model_analysis,
  #     #                  train_steps=5e5,
  #     #                  from_db_path=FROM_DB_PATH)
  #     # final_model_dir = tune_best_model(experiment_name=agentname, analysis=best_model_analysis,
  #     # with_pbt=True).best_checkpoint
  #     print('exiting...')
  #     # print(f'Trained model weights and checkpoints are stored in {final_model_dir}')
  #   else:
  #     pass
  #     # # todo include num_players to sql query
  #     num_players = 3
  #     # agentname = 'VanDenBerghAgent'
  #     config = {'agent': FlawedAgent,
  #               'lr': 2e-3,
  #               'num_hidden_layers': 1,
  #               'layer_size': 64,
  #               'batch_size': BATCH_SIZE,  # tune.choice([4, 8, 16, 32]),
  #               'num_players': num_players,
  #               'agent_config': {'players': num_players}
  #               }
  #     train_eval(config,
  #                env_config=hanabi_config,
  #                conn=None,
  #                checkpoint_dir=None,
  #                from_db_path=db_path,
  #                target_table='pool_of_state_dicts',
  #                log_interval=LOG_INTERVAL,
  #                eval_interval=EVAL_INTERVAL,
  #                num_eval_states=NUM_EVAL_STATES,
  #                max_train_steps=np.inf,
  #                use_ray=False)
  #

if __name__ == '__main__':
  players = 3
  hanabi_config = {
    "colors":
      3,
    "ranks":
      5,
    "players":
      players,
    "hand_size":
      4 if players in [4, 5] else 5,
    # hand size is derived from number of players
    "max_information_tokens":
      8,
    "max_life_tokens":
      3,
    "observation_type":
      1  # pyhanabi.AgentObservationType.CARD_KNOWLEDGE.value
  }
  main(hanabi_config)
