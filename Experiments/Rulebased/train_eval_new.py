import os
import pickle
import traceback

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import torch
from torch import optim
from Experiments.Rulebased.config_new import hanabi_config, ray_config, database_size, pool_size, GRACE_PERIOD, \
  RAY_NUM_SAMPLES, MAX_T, KEEP_CHECKPOINTS_NUM, VERBOSE, log_interval, eval_interval, n_states_for_evaluation
from utils import stringify_env_config, get_observation_length, get_max_actions, to_int
from data import maybe_create_and_populate_database
from datasets import PoolOfStates
from cl2 import AGENT_CLASSES, StateActionCollector
import model


def eval_fn(env_config, net, eval_loader, criterion, target_agent, num_states):
  # load pickled observations and get vectorized and compute action and eval with that
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


def parse_batch(b, pyhanabi_as_bytes):
  ret = []
  for obs in b:
    d = dict(obs)
    if pyhanabi_as_bytes:
      d['pyhanabi'] = pickle.loads(obs['pyhanabi'])
    ret.append(d)
  return ret


def train_eval(config,  # used by ray
               env_config,
               agentcls,
               pool_size,
               from_db_path,
               log_interval,
               eval_interval,
               num_eval_states,
               pyhanabi_as_bytes=True,
               checkpoint_dir=None,
               data_dir=None, ):
  lr = config['lr']
  num_hidden_layers = config['num_hidden_layers']
  layer_size = config['layer_size']
  batch_size = config['batch_size']

  agent = agentcls({'players': env_config['players']})

  net = model.get_model(observation_size=get_observation_length(env_config),
                        num_actions=get_max_actions(env_config),
                        num_hidden_layers=num_hidden_layers,
                        layer_size=layer_size)

  # Create Pool of States of first n rows from database for training
  trainloader_fn = partial(PoolOfStates(from_db_path).get_eagerly, n_rows=pool_size,
                           pyhanabi_as_bytes=True,
                           batch_size=batch_size,
                           pick_at_random=False,  # random_seed=42
                           )
  testloader = StateActionCollector(hanabi_game_config=env_config,
                                    agent_classes=AGENT_CLASSES)

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

  while True:
    trainloader = trainloader_fn()  # train sampling first pool_size states from database
    try:
      for batch_raw in trainloader:
        batch = parse_batch(batch_raw, pyhanabi_as_bytes)
        actions = torch.LongTensor([to_int(env_config, agent.act(obs)) for obs in batch])
        vectorized = torch.FloatTensor([obs['vectorized'] for obs in batch])
        optimizer.zero_grad()
        outputs = net(vectorized).reshape(batch_size, -1)
        loss = criterion(outputs, actions)

        loss.backward()
        optimizer.step()

        if it % log_interval == 0:
          print(f'Iteration {it}...')
        if it % eval_interval == 0:
          loss, acc = eval_fn(env_config, net=net, eval_loader=testloader, criterion=criterion,
                              target_agent=agent,
                              num_states=num_eval_states)
          moving_acc += acc
          eval_it += 1
          # tune.report(training_iteration=it, loss=loss, acc=moving_acc / eval_it)
          tune.report(training_iteration=it, loss=loss, acc=acc)
          # checkpoint frequency may be handled by ray if we remove checkpointing here
          with tune.checkpoint_dir(step=it) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save((net.state_dict, optimizer.state_dict), path)
        # if it > max_train_steps:
        #   return
        it += 1
    except Exception as e:
      if isinstance(e, StopIteration):
        epoch += 1
        continue
      else:
        print(e)
        print(traceback.print_exc())
        raise e


def select_best_model(env_config,
                      ray_config,
                      db_path,
                      pool_size,
                      agentname,
                      agentcls, ):
  ckpt_dir = f'./{agentname}_{stringify_env_config(hanabi_config)}/'
  # configure via config_new.py
  train_fn = tune.with_parameters(train_eval,
                                  checkpoint_dir=ckpt_dir,
                                  agentcls=agentcls,
                                  pool_size=pool_size,
                                  from_db_path=db_path,
                                  env_config=env_config,
                                  log_interval=log_interval,
                                  eval_interval=eval_interval,
                                  num_eval_states=n_states_for_evaluation,
                                  pyhanabi_as_bytes=True)
  scheduler = ASHAScheduler(time_attr='training_iteration', grace_period=GRACE_PERIOD, max_t=MAX_T)
  analysis = tune.run(train_fn,
                      metric='acc',
                      mode='max',
                      config=ray_config,
                      name=agentname,
                      num_samples=RAY_NUM_SAMPLES,
                      keep_checkpoints_num=KEEP_CHECKPOINTS_NUM,
                      verbose=VERBOSE,
                      scheduler=scheduler,
                      progress_reporter=CLIReporter(metric_columns=["loss", "acc", "training_iteration"]),
                      )
  best_trial = analysis.get_best_trial("acc", "max")
  print(best_trial.config)
  print(best_trial.checkpoint)
  return analysis


def main():
  db_path = f'./database_{stringify_env_config(hanabi_config)}.db'
  # check if database exists for corresponding config, otherwise create and insert 500k states [takes a long time]
  maybe_create_and_populate_database(db_path,
                                     hanabi_config,
                                     database_size)

  # train models for each agent on pool_size states from database and evaluate collecting online games
  assert database_size > pool_size, 'not enough states in database for training with pool_size'
  for agentname, agentcls in AGENT_CLASSES.items():
    best_model_analysis = select_best_model(env_config=hanabi_config,
                                            ray_config=ray_config,
                                            db_path=db_path,
                                            pool_size=pool_size,
                                            agentname=agentname,
                                            agentcls=agentcls)


if __name__ == '__main__':
  # Goal is to find a reasonable lower bound on pool_size
  # todo parameterize with pool size
  main()
