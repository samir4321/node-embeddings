# tree_utils.py
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence


def transition_probs(tree, p, q):
    """
    build normalized transition probabilities
    P(t, v, x) = normalized probability of x as next node, given
    current node is v and previous node was t

    :param p: 1/p = unnormalized probability of return to previous node
    :param q: 1 / q = unnormalized probability of random exploration of neighbor node of distance 2 or greater from
    previous node
        1 is unnormalized probability of transition to node of distance 1 from previous node
    """
    adj_list = tree_to_adjlist(tree)
    #print(f'adjacency list: {adj_list}')
    tr_probs = {}
    for t in adj_list.keys():
        for v in adj_list[t]:
            unnormed_probs = {}
            for x in adj_list[v]:
                # if x equals t -> 1 / p
                if x == t:
                    unnormed_probs[x] = 1. / p
                # if x of distance 1 from t -> 1
                elif x in adj_list[t]:
                    unnormed_probs[x] = 1.
                else:
                    unnormed_probs[x] = 1. / q
            sm = sum(unnormed_probs.values())
            for x in adj_list[v]:
                tr_probs[(t, v, x)] = unnormed_probs[x] / sm
    return tr_probs


def generate_random_walks(tree, p, q, walk_length, nwalks):
   """
   :param tree:
   :param walk_length:
   :param p:
   :param q:
   :return:
   """
   tr_probs = transition_probs(tree, p, q)
   adj_list = tree_to_adjlist(tree)
   walks = []
   for nd in adj_list.keys():  # could run in parallel
    for nw in range(nwalks):
        walk = [nd, np.random.choice(list(adj_list[nd]))] # first two nodes in walk
        for i in range(walk_length - 2): # rest of walk
            possible_x = list(adj_list[walk[i + 1]])
            next_transition_probs = [tr_probs[(walk[i], walk[i+1], x)] for x in possible_x]
            next_node = np.random.choice(possible_x, p=next_transition_probs)
            walk.append(next_node)
        walks.append(''.join(walk))
   return walks

def generate_training_data(node_set, random_walks, window_size, input_dim, batch_size):
    # TODO(add negative sampling)
    maxlen = window_size * 2
    node_index = dict((v, k) for k, v in enumerate(node_set))
    np.random.shuffle(random_walks)
    for walk in random_walks:
        walk_indices = [node_index[nd] for nd in walk]
        l = len(walk_indices)
        xb, yb = [], []
        for ix, node in enumerate(walk_indices):
            contexts = []
            labels = []
            s = ix - window_size
            e = ix + window_size + 1
            if s >= 0:
                window = [walk_indices[i] for i in range(s, e) if 0 <= i < l and i != ix]
                contexts.append(window)
                labels.append(node)
                x = sequence.pad_sequences(contexts, maxlen=maxlen)
                y = to_categorical(labels, input_dim)
                xb.append(x)
                yb.append(y)
            if len(xb) > 0 and len(xb) % batch_size == 0:
                xb = np.array(xb).reshape((batch_size, maxlen))
                yb = np.array(yb).reshape((batch_size, input_dim))
                yield xb, yb
                xb, yb = [], []


def shuffle_together(arr1, arr2):
    assert (len(arr1) == len(arr2))
    p = np.random.permutation(len(arr1))
    return arr1[p], arr2[p]


def tree_to_pairs(d, current=None):
    pairs = []
    if d is None:
        return pairs
    if current is None:
        for child in d.keys():
            new_pairs = tree_to_pairs(d[child], child)
            pairs = pairs + new_pairs
    else:
        for child in d.keys():
            pairs.append((child, current))
            new_pairs = tree_to_pairs(d[child], child)
            pairs = pairs + new_pairs
    return pairs


def tree_to_adjlist(d):
    pairs = tree_to_pairs(d, None)
    adj_list = {}
    for a, b in pairs:
        if a not in adj_list.keys():
            adj_list[a] = set()
        adj_list[a].add(b)
        if b not in adj_list.keys():
            adj_list[b] = set()
        adj_list[b].add(a)
    return adj_list

