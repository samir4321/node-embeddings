# cbow model word2vec
import numpy as np
np.random.seed(13)

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
import gensim

path = get_file('alice.txt', origin='http://www.gutenberg.org/files/11/11-0.txt')
full_corpus = open(path).readlines()
n_samples = len(full_corpus)#300
corpus = full_corpus[:n_samples]
corpus = [sentence for sentence in corpus if sentence.count(' ') >= 2]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
corpus = tokenizer.texts_to_sequences(corpus)
nb_samples = sum(len(s) for s in corpus)
V = len(tokenizer.word_index) + 1
embedding_dim = 100
window_size = 3
batch_size = 10
nepochs = 100
n_iters = int(nepochs * n_samples / batch_size)
save_freq = 5#

def generate_data(corpus, window_size, V):
    maxlen = window_size * 2
    for words in corpus:
        L = len(words)
        xb, yb = [], []
        for index, word in enumerate(words):
            contexts = []
            labels = []
            s = index - window_size
            e = index + window_size + 1
            contexts.append(
                [words[i] for i in range(s, e) if 0 <= i < L and i != index])
            labels.append(word)

            x = sequence.pad_sequences(contexts, maxlen=maxlen)
            y = to_categorical(labels, V)
            xb.append(x)
            yb.append(y)
            if len(xb) > 0 and len(xb) % batch_size == 0:
                xb = np.array(xb).reshape((batch_size, maxlen))
                yb = np.array(yb).reshape((batch_size, V))
                yield xb, yb
                xb, yb = [], []
            #yield (x, y)

def build_model():
    cbow = Sequential()
    cbow.add(Embedding(input_dim=V, output_dim=embedding_dim, input_length=window_size * 2))
    cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim,)))
    cbow.add(Dense(V, activation='softmax'))
    cbow.compile(loss='categorical_crossentropy', optimizer='adam')
    cbow.summary()
    return cbow


def train_model(cbow):
    for ite in range(n_iters):
        loss = 0.
        for x, y in generate_data(corpus, window_size, V):
            loss += cbow.train_on_batch(x, y)
        print(ite, loss)
        if ite > 0 and ite % save_freq == 0:
            save_model(cbow)
            print('saving model ...')

def save_model(cbow):
    f = open('vectors.txt' ,'w')
    f.write('{} {}\n'.format(V - 1, embedding_dim))
    vectors = cbow.get_weights()[0]
    for word, i in tokenizer.word_index.items():
        str_vec = ' '.join(map(str, list(vectors[i, :])))
        f.write('{} {}\n'.format(word, str_vec))
    f.close()

def run_similarity():
    w2v = gensim.models.KeyedVectors.load_word2vec_format('./vectors.txt', binary=False)
    for word, sim_score in w2v.most_similar(positive=['quiet']):
        print(f'word: {word} similarity score: {sim_score}')

def main():
   #cbow = build_model()
   #train_model(cbow)
   #save_model(cbow)
    run_similarity()

main()
