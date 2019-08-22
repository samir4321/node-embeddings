# tree_hash
import hashlib

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








