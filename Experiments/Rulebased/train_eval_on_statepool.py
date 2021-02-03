import os
from cl2 import AGENT_CLASSES, StateActionCollector
from data import maybe_create_and_populate_database
from config import hanabi_config, database_size, pool_size, num_hidden_layers, layer_sizes, \
  n_states_for_evaluation, hyperparam_grid, log_interval, eval_interval, max_train_steps, ckpt_interval
from utils import stringify_env_config, get_observation_length, get_max_actions, to_int
import model
from datasets import PoolOfStates
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import pickle
import traceback
from time import time
from functools import partial
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

# [x]. set game configuration
# [x]. if it does not exist: create and populate database for current game config
# 2. loop hyperparams and models:
# 3. train and eval using_num_states many rows from the database the current model&hps
# 4. checkpointing
# 5. earlystopping
# 6. tensorboards
# 7. report best performing model


class TrainEvalRunner:
  def __init__(self, agentcls,
               net,
               hparams,
               trainloader_fn,
               testloader,
               env_config,
               ckpt_dir=None):
    self.agentcls = agentcls
    self.net = net
    self.hparams = hparams
    self.trainloader_fn = trainloader_fn
    self.testloader = testloader
    self.env_config = env_config
    self.ckpt_dir = ckpt_dir
    self.lr = hparams['lr'][0]  # todo apply gridsearch here
    self.batch_size = hparams['batch_size']
    self.agent = agentcls({'players': env_config['players']})
    self.writer = SummaryWriter(log_dir=self.ckpt_dir,)

  @staticmethod
  def _parse_batch(b, pyhanabi_as_bytes=True):
    ret = []
    for obs in b:
      d = dict(obs)
      if pyhanabi_as_bytes:
        d['pyhanabi'] = pickle.loads(obs['pyhanabi'])
      ret.append(d)
    return ret

  def eval_fn(self, criterion):
    # load pickled observations and get vectorized and compute action and eval with that
    start = time()
    observations_pickled = self.testloader.collect(num_states_to_collect=n_states_for_evaluation,
                                                   keep_obs_dict=True,
                                                   keep_agent=False)
    assert len(
      observations_pickled) == n_states_for_evaluation, f'len(observations_pickled)={len(observations_pickled)} and num_states = {n_states_for_evaluation}'
    # print(f'Collecting eval states took {time() - start} seconds')
    correct = 0
    running_loss = 0
    with torch.no_grad():
      for obs in observations_pickled:
        observation = pickle.loads(obs)
        action = torch.LongTensor([to_int(self.env_config, self.agent.act(observation))])
        prediction = self.net(torch.FloatTensor(observation['vectorized'])).reshape(1, -1)
        # loss
        running_loss += criterion(prediction, action)
        # accuracy
        correct += torch.sum(torch.max(prediction, 1)[1] == action)

      return 100 * running_loss / n_states_for_evaluation, 100 * correct.item() / n_states_for_evaluation

  def maybe_log(self, it, log_interval):
    """ Possibly do more complex logging here """
    if it % log_interval == 0:
      pass  # print(f'Iteration {it}...')

  def maybe_ckpt(self, it, optimizer):
    if it % ckpt_interval == 0:
      if self.ckpt_dir :
        print(f'saving model to {self.ckpt_dir}')
        path = os.path.join(self.ckpt_dir, f'checkpoint_+{it}')
        torch.save((self.net.state_dict, optimizer.state_dict), path)

  def maybe_stop_early(self, it):
    """ Possibly do more complex early stopping here """
    if it > max_train_steps:
      return

  def eval(self, it, criterion, optimizer):
    eval_loss, acc = self.eval_fn(criterion)
    # print(f'Loss at iteration {it} is {loss}, and accuracy is {moving_acc / eval_it} %')
    print(f'Eval loss: {eval_loss}, accuracy: {acc} %')
    self.writer.add_scalar('Loss/test', eval_loss, it)
    self.writer.add_scalar('Accuracy/test', acc, it)

    return eval_loss, acc

  def train_eval(self):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
    it = 0
    epoch = 1
    moving_acc = 0
    eval_it = 0
    trainloader = self.trainloader_fn()
    while True:
      try:
        for batch_raw in trainloader:  # list of dictionaries
          batch = self._parse_batch(batch_raw)
          actions = torch.LongTensor([to_int(self.env_config, self.agent.act(obs)) for obs in batch])
          vectorized = torch.FloatTensor([obs['vectorized'] for obs in batch])
          optimizer.zero_grad()
          outputs = self.net(vectorized).reshape(self.batch_size, -1)
          train_loss = criterion(outputs, actions)

          train_loss.backward()
          optimizer.step()
          self.maybe_log(it, log_interval)
          if it % eval_interval == 0:  # maybe eval, write_summary for tensorboard
            print(f'Iteration: {it}\n ')
            print(f'Train loss: {train_loss}')
            eval_loss, acc = self.eval(it, criterion, optimizer)
            moving_acc += acc
            eval_it += 1
          self.maybe_ckpt(it, optimizer)  # maybe save model and optim
          self.maybe_stop_early(it)
          it += 1
      except Exception as e:
        if isinstance(e, StopIteration):
          # pool of states has been read fully, start over
          trainloader = self.trainloader_fn()  # returns first pool_size rows of database
          epoch += 1
          print('SADHKLJLAHSHDASHLJKDAJHSKLJHLKDSA')
          continue

        else:
          print(traceback.print_exc())
          exit(1)


def find_and_train_eval_best_model(agentcls, from_db_path, hparams, agentname,
                                   env_config, pool_size):
  """ Tune models chosen after manual search """
  nets = [model.get_model(observation_size=get_observation_length(env_config),
                          num_actions=get_max_actions(env_config),
                          num_hidden_layers=num_hidden_layers,
                          layer_size=layer_size) for layer_size in layer_sizes]
  # Create Pool of States of first n rows from database for training
  trainloader_fn = partial(PoolOfStates(from_db_path).get_eagerly, n_rows=pool_size,
                                                       pyhanabi_as_bytes=True,
                                                       batch_size=hparams['batch_size'],
                                                       pick_at_random=False, # random_seed=42
                                                       )
  # testloader does not get data from database, but collects it live
  testloader = StateActionCollector(hanabi_game_config=env_config,
                                    agent_classes=AGENT_CLASSES)

  # run training and evaluation for each net using pool_size states and save checkpoints to ckpt_dir
  ckpt_dir = f'./{agentname}_{stringify_env_config(hanabi_config)}/'
  [TrainEvalRunner(agentcls=agentcls,
                   net=net,
                   hparams=hparams,
                   trainloader_fn=trainloader_fn,
                   testloader=testloader,
                   env_config=env_config,
                   ckpt_dir=ckpt_dir).train_eval() for net in nets]


def main():
  db_path = f'./database_{stringify_env_config(hanabi_config)}.db'
  # check if database exists for corresponding config,
  # otherwise create and insert 500k states [takes a long time]
  maybe_create_and_populate_database(db_path,
                                     hanabi_config,
                                     database_size)

  # train models for each agent on pool_size states from database and evaluate collecting online games
  assert database_size > pool_size, 'not enough states in database for training with pool_size'
  for agentname, agentcls in AGENT_CLASSES.items():
    find_and_train_eval_best_model(agentcls=agentcls,
                                   from_db_path=db_path,
                                   hparams=hyperparam_grid,
                                   agentname=agentname,
                                   env_config=hanabi_config,
                                   pool_size=pool_size)


if __name__ == '__main__':
  # todo gridsearchcv
  # Goal is to find a reasonable lower bound on pool_size
  main()
