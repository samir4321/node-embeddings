# tree_embed.py
d1 = {'A': {'B': {'C': None, 'D': None}}}
d2 = {'A': {'B': {'C': None, 'E': None}}}
d3 = {'A': {'Q': {'C': None, 'D': None}}}


def random_walk(d, node, nsteps):
    """
     perform a random walk on the given tree (treated as a graph)
     of n steps, beginning from the given node
    """
    pass

def get_node(d, node):
    for c in d.keys():
        if c == node:
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
    print(pairs)
    adj_list = {}
    for a, b in pairs:
        if a not in adj_list.keys():
            adj_list[a] = set()
        adj_list[a].add(b)
        if b not in adj_list.keys():
            adj_list[b] = set()
        adj_list[b].add(a)
    return adj_list

print(tree_to_adjlist(d1))