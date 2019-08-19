# endpoint_flow_embed.py


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