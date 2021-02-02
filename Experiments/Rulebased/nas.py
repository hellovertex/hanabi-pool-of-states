import torch
import numpy as np

from data import IterableStatesCollection, AGENT_CLASSES
from model import gen_model


def get_batch_jacobian(net, x, target):
    net.zero_grad()

    x.requires_grad_(True)
    # res = net(x)

    y = net(x)
    # compute [dyi/dxj] * [1,...,1] = J * v^T
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob, target.detach()


def eval_score(jacob):
    corrs = np.corrcoef(jacob)
    v, _ = np.linalg.eigh(corrs)
    k = 1e-5
    return -np.sum(np.log(v + k) + 1. / (v + k))


BATCH_SIZE = 64
train_loader = IterableStatesCollection(AGENT_CLASSES,
                                        num_players=3,
                                        agent_id="InternalAgent",
                                        batch_size=BATCH_SIZE)

# x, y = next(iter(train_loader))
# print(x.shape, y.shape)
num_batches_for_jacob = 100
cum_scores = None
besties = []
for _ in range(30):
    scores = []
    for net in gen_model(959, 30):
        # for _ in range(num_batches_for_jacob):
        x, y = next(iter(train_loader))
        jacob, target = get_batch_jacobian(net, x, y)
        jacob = jacob.reshape(jacob.size(0), -1).cpu().numpy()

        try:
            s = eval_score(jacob)
        except Exception as e:
            print(e)
            s = np.nan

        scores.append(s)
    if cum_scores is not None:
        cum_scores = np.array(scores) + cum_scores
    else:
        cum_scores = np.array(scores)

    print(scores)
    besties.append(np.argmax(np.array(scores)))
    print(besties)
print('winner is', np.argmax(cum_scores))