# svd_node_embed.py
"""
 embed nodes using nearest neighbors
 from random walk
"""
import numpy as np
import time

import tree_utils

DEFAULT_WALK_LENGTH = 20  # length random walks to be generated from a given node
P = 1.0  # return hyperparameter
Q = 0.5  # 1.0 inout hyperparameter


def sample_graphs():
    d1 = {'A': {'B': {'C': None, 'D': {"E": None, "F": None}}}}
    d2 = {'A': {'Q': {'C': None, 'D': {"E": None, "F": None}}}}
    samples = [d1, d2]
    return samples


def svd_node_embed(trees, embed_dim, nwalks=1000):
    tree_hashes = {}
    for tree in trees:
        tree_hashes[tree_utils.get_hash(tree)] = tree
    unique_trees = tree_hashes.values()
    all_walks = []
    for tree in unique_trees:
        walks = tree_utils.generate_random_walks(tree, P, Q, DEFAULT_WALK_LENGTH,
                                      nwalks=nwalks)
        all_walks += walks
    np.random.shuffle(all_walks)

    neighbors = {}
    ix = {}
    i = 0
    for walk in all_walks:
        for j, node in enumerate(walk):
            if not node in ix.keys():
                ix[node] = i
                neighbors[i] = []
                i += 1
        for j, node in enumerate(walk):
            if j > 0:
                neighbors[ix[node]].append(ix[walk[j - 1]])
            if j < len(walk) - 1:
                neighbors[ix[node]].append(ix[walk[j + 1]])
    X = np.zeros((i, i))
    for row in range(i):
        for col in range(i):
            X[row, col] = neighbors[row].count(col)
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    proj_u = U[:, :embed_dim]
    node_embed = {}
    for key in ix.keys():
        node_embed[key] = proj_u[ix[key]]
    return node_embed


def main():
    trees = sample_graphs()
    start = time.time()
    embedding = svd_node_embed(trees, 2)
    end = time.time()
    np.set_printoptions(precision=3, suppress=True)
    print("-" * 20)
    print(f'ran node svd on {len(trees)} trees in {(end - start):.2f} seconds')
    print('-' * 20)
    for node, v in embedding.items():
        print(f'node: {node} embedding: {v}')

if __name__ == "__main__":
    main()

