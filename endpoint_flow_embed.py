# endpoint_flow_embed.py
"""
ideas

  classify kind of graph in embedding (one region cliques, the other single sequences,
  the other fan out, etc.)


  https://www-cs.stanford.edu/people/jure/pubs/graphrepresentation-ieee17.pdf


  link prediction: probability of a connection between nodes given existing data,
  use higher-order probabilities (means non-zero probability even if no link in the empirical data)
    HOPE algorithm

  if endpoint connections can be sparse/it's expensive to compute the empirical probability A -> B,
  P(B|A) from the data for all nodes, use graph embeddings


  semi-supervised learning of node classes: given labels for only some nodes, learn labels for other nodes


  measure drift/sudden changes in graph structure

  suppose a short sudden change in the graph structure between t1 and t2. detect and ideally interpret
  the change. for example, 'over t1 to t2 node A connected to nodes Q, P, S, which is highly unusual'

"""

ENDPOINTS = [hex(i) for i in range(50)]
MAX_DEPTH = 8


def api_graph():
    """
    service graph with two separate subgraphs

    """

    d = {}
    d['A'] = ['B', 'Q']
    d['B'] = ['C', 'D', 'E']
    d['Q'] = ['C', 'E']

    d['N'] = ['T', 'V']
    d['T'] = ['W']
    return d


def build_graph_from_data():
    # graph of all possible connections gleaned from data
    # here assumed P(B|A) > 0 empirically => A -> B in graph
    data = [
        'ABC',
        'ABE',
        'ABD',
        'AQC',
        'AQE',
        'NT',
        'TW',
        'NV'
    ]
    d = {}
    for datum in data:
        for i, c in enumerate(datum):
            if i < (len(datum) - 1):
                if c not in d.keys():
                    d[c] = []
                next_c = datum[i+1]
                if next_c not in d[c]:
                    d[c].append(datum[i+1])
                    d[c] = sorted(d[c])
    return d

def is_allowed(d, src, dst):
    return (src in d.keys() and dst in d[src])


d = build_graph_from_data()
assert(is_allowed(d, 'A', 'B'))
assert(not is_allowed(d, 'A', 'T'))





