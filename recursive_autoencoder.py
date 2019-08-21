# recursive_autoencoder.py
"""
 recursive autoencoder for tree structures
 recursion defined as

    parent = affine transformation of direct sum of two randomly chosen children

"""

import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Lambda
from tensorflow.keras.utils import get_file
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

d1 = {'A': {'B': {'C': None, 'D': None}}}
d2 = {'A': {'B': {'C': None, 'E': None}}}
d3 = {'A': {'Q': {'C': None, 'D': None}}}

dim = 100

def build_model():
    recursive_ae = Sequential()
