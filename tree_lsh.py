# tree_lsh.py
"""
 (non) locally-sensitive hashing of tree structures

    A
    |
    B
   / \
  C   D
"""
import hashlib

d1 = {'A': {'B': {'C': None, 'D': None}}}
d2 = {'A': {'B': {'C': None, 'E': None}}}
d3 = {'A': {'Q': {'C': None, 'D': None}}}


def build_merkle_tree(d):
    pass


def get_hash(d):
    """
    A simple tree hash defined as the merkle tree root
    use case: given a new incoming tree (trace), determine if/statistics
    of how much it has been seen before
    """
    node_hashes = []
    for key in d.keys():
        node = d[key]
        if node is None:
            hashv = hashlib.sha256(key.encode('utf-8')).hexdigest()
            return hashv
        node_hash = get_hash(node)
        node_hashes.append(node_hash)
    concat_hash = ''.join(node_hashes)
    return hashlib.sha256(concat_hash.encode('utf-8')).hexdigest()


print(get_hash(d1))





