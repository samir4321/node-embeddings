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
from tensorflow.keras import optimizers
import gensim

import tree_utils


DEFAULT_WALK_LENGTH = 20  # length random walks to be generated from a given node
P = 1.0  # return hyperparameter
Q = 0.5 #1.0  # inout hyperparameter
NODE_SET = string.ascii_uppercase[:6]
INPUT_DIM = len(NODE_SET)  # dimension of input space (number of possible nodes)
EMBEDDING_DIM = 2   # dimension for embedding space, should be ~ input_dim ^ 0.25
WINDOW_SIZE = 3  # number of nodes each of left and right to include in window taken from random walk
BATCH_SIZE = 5  # training batch_size

optimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


def sample_graphs():
    d1 = {'A': {'B': {'C': None, 'D': {"E": None, "F": None}}}}
    #d2 = {'A': {'B': {'C': None, 'D': None}}}
    #d3 = {'A': {'Q': {'C': None, 'D': None}}}
    return [d1] * 100#[d1] * 100 + [d3] * 1


def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=INPUT_DIM, output_dim=EMBEDDING_DIM,
                       input_length=WINDOW_SIZE * 2))
    model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(EMBEDDING_DIM,)))
    model.add(Dense(INPUT_DIM, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.summary()
    return model


def train_model(node2vec, n_epochs, save_freq):
    all_walks = []
    for tree in sample_graphs():
        walks = tree_utils.generate_random_walks(tree, P, Q, DEFAULT_WALK_LENGTH,
                                      nwalks=2)
        all_walks += walks
    np.random.shuffle(all_walks)
    print(f'Training on windows for {len(all_walks)} random walks')
    for epoch in range(n_epochs):
        loss = 0.
        for xb, yb in tree_utils.generate_training_data(NODE_SET, all_walks, WINDOW_SIZE, INPUT_DIM,
                                             BATCH_SIZE):
            batch_loss = node2vec.train_on_batch(xb, yb)
            loss += batch_loss
        print(f'epoch: {epoch} epoch loss: {loss}')
        if epoch > 0 and epoch % save_freq == 0:
            save_model(node2vec)


def save_model(node2vec):
    print('saving model ...')
    f = open('./node2vec_vectors.txt' ,'w')
    f.write('{} {}\n'.format(INPUT_DIM - 1, EMBEDDING_DIM))
    vectors = node2vec.get_weights()[0]
    node_index = dict((nd, ix) for ix, nd in enumerate(NODE_SET))
    for nd, i in node_index.items():
        str_vec = ' '.join(map(str, list(vectors[i, :])))
        f.write('{} {}\n'.format(nd, str_vec))
    f.close()


def main():
    node2vec = build_model()
    n_epochs = 100
    save_freq = 1
    train_model(node2vec, n_epochs, save_freq)


if __name__ == "__main__":
    main()

