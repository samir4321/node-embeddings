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
    print(d)

build_graph_from_data()