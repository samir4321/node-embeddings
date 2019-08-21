# node2vec.py
"""
 an embedding of graph nodes

 note- algorithm allows edge weights for affecting
 random walk probabilities, but for the time being will
 assume all equal.
"""
import numpy as np
import random
import string
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
import gensim


DEFAULT_WALK_LENGTH = 10  # number of random walks to be generated from a given node
P = 1.0  # return hyperparameter
Q = 1.0  # inout hyperparameter
INPUT_DIM = 26  # dimension of input space (number of possible nodes)
EMBEDDING_DIM = 10  # dimension for embedding space
WINDOW_SIZE = 2  # number of nodes each of left and right to include in window taken from random walk
BATCH_SIZE = 5  # training batch_size

# tokenizer.fit_on_texts(corpus)
# corpus = tokenizer.texts_to_sequences(corpus)
# nb_samples = sum(len(s) for s in corpus)
# V = len(tokenizer.word_index) + 1
# embedding_dim = 100
# window_size = 3
# batch_size = 10
# nepochs = 100
# n_iters = int(nepochs * n_samples / batch_size)
# save_freq = 5


def sample_graphs():
    d1 = {'A': {'B': {'C': None, 'D': {"F": None, "H": None}}}}
    #d2 = {'A': {'B': {'C': None, 'D': None}}}
    d3 = {'A': {'Q': {'C': None, 'D': None}}}
    return [d1] * 100 + [d3] * 1


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
        walk = [nd, random.choice(list(adj_list[nd]))] # first two nodes in walk
        for i in range(walk_length - 2): # rest of walk
            possible_x = list(adj_list[walk[i + 1]])
            next_transition_probs = [tr_probs[(walk[i], walk[i+1], x)] for x in possible_x]
            next_node = np.random.choice(possible_x, p=next_transition_probs)
            walk.append(next_node)
        walks.append(''.join(walk))
   return walks


def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=INPUT_DIM, output_dim=EMBEDDING_DIM,
                       input_length=WINDOW_SIZE * 2))
    model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(EMBEDDING_DIM,)))
    model.add(Dense(INPUT_DIM, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model


def generate_training_data(random_walks, window_size, input_dim, batch_size):
    maxlen = window_size * 2
    node_index = dict((v, k) for k, v in enumerate(string.ascii_uppercase))
    random.shuffle(random_walks)
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


# def generate_data(corpus, window_size, V):
#     maxlen = window_size * 2
#     for words in corpus:
#         L = len(words)
#         xb, yb = [], []
#         for index, word in enumerate(words):
#             contexts = []
#             labels = []
#             s = index - window_size
#             e = index + window_size + 1
#             contexts.append(
#                 [words[i] for i in range(s, e) if 0 <= i < L and i != index])
#             labels.append(word)
#
#             x = sequence.pad_sequences(contexts, maxlen=maxlen)
#             y = to_categorical(labels, V)
#             xb.append(x)
#             yb.append(y)
#             if len(xb) > 0 and len(xb) % batch_size == 0:
#                 xb = np.array(xb).reshape((batch_size, maxlen))
#                 yb = np.array(yb).reshape((batch_size, V))
#                 yield xb, yb
#                 xb, yb = [], []
#             #yield (x, y)

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


def train_model(node2vec, n_epochs, save_freq):
    all_walks = []
    for tree in sample_graphs():
        walks = generate_random_walks(tree, P, Q, DEFAULT_WALK_LENGTH,
                                      nwalks=10)
        all_walks += walks
    np.random.shuffle(all_walks)
    print(f'Training on windows for {len(all_walks)} random walks')
    for epoch in range(n_epochs):
        loss = 0.
        for xb, yb in generate_training_data(all_walks, WINDOW_SIZE, INPUT_DIM,
                                             BATCH_SIZE):
            batch_loss = node2vec.train_on_batch(xb, yb)
            loss += batch_loss
        print(f'epoch: {epoch} epoch loss: {loss}')
        if epoch > 0 and epoch % save_freq == 0:
            save_model(node2vec)



def save_model(node2vec):
    print('saving model ...')
    pass
    # f = open('node2vec_vectors.txt' ,'w')
    # f.write('{} {}\n'.format(INPUT_DIM - 1, EMBEDDING_DIM))
    # vectors = node2vec.get_weights()[0]
    # for word, i in tokenizer.word_index.items():
    #     str_vec = ' '.join(map(str, list(vectors[i, :])))
    #     f.write('{} {}\n'.format(word, str_vec))
    # f.close()


def main():
    samples = sample_graphs()
    node2vec = build_model()
    n_epochs = 100
    save_freq = 1
    train_model(node2vec, n_epochs, save_freq)
    # tree = samples[0]
    # tr_probs = transition_probs(tree, P, Q)
    # #print(tr_probs)
    # walks = generate_random_walks(tree, P, Q, DEFAULT_WALK_LENGTH, nwalks=100)
    # for xb, yb in generate_training_data(walks, WINDOW_SIZE, INPUT_DIM, BATCH_SIZE):
    #     print((xb, yb))


main()

