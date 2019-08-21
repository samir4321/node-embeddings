# node2vec.py
"""
 an embedding of graph nodes

 note- algorithm allows edge weights for affecting
 random walk probabilities, but for the time being will
 assume all equal.
"""
import numpy as np
import random

DEFAULT_WALK_LENGTH = 20  # number of random walks to be generated from a given node
P = 1.0  # return hyperparameter
Q = 1.0  # inout hyperparameter


def sample_graphs():
    d1 = {'A': {'B': {'C': None, 'D': {"F": None, "H": None}}}}
    d2 = {'A': {'B': {'C': None, 'E': None}}}
    d3 = {'A': {'Q': {'C': None, 'D': None}}}
    return [d1, d2, d3]


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
    print(f'adjacency list: {adj_list}')
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
   for nd in adj_list.keys(): # could run in parallel
    for nw in range(nwalks):
        walk = [nd, random.choice(list(adj_list[nd]))] # first two nodes in walk
        for i in range(walk_length - 2): # rest of walk
            possible_x = list(adj_list[walk[i + 1]])
            next_transition_probs = [tr_probs[(walk[i], walk[i+1], x)] for x in possible_x]
            next_node = np.random.choice(possible_x, p=next_transition_probs)
            walk.append(next_node)
        walks.append(walk)
   return walks


def build_model():
    pass


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


def main():
    samples = sample_graphs()
    tree = samples[0]
    tr_probs = transition_probs(tree, P, Q)
    #print(tr_probs)
    walks = generate_random_walks(tree, P, Q, DEFAULT_WALK_LENGTH, nwalks=100)
    for w in walks:
        print(w)


main()