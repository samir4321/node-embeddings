# node2vec.py
"""
 an embedding of graph nodes

 note- algorithm allows edge weights for affecting
 random walk probabilities, but for the time being will
 assume all equal.
"""
DEFAULT_WALK_LENGTH = -1 # number of random walks to be generated from a given node
P = -1 # return hyperparameter
Q = -1 # inout hyperparameter


def sample_graphs():
    d1 = {'A': {'B': {'C': None, 'D': None}}}
    d2 = {'A': {'B': {'C': None, 'E': None}}}
    d3 = {'A': {'Q': {'C': None, 'D': None}}}
    return [d1, d2, d3]

def generate_random_walk_data(graphs, walk_length, p, q):

    pass

def build_model():
    pass

def main():
    pass

