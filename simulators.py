# simulators.py
"""
 data and embedding simulators motivated by
 use cases

"""
import numpy as np
import tree_utils
import svd_node_embed


def service_down_simulator():
    print(service_down_diagram())
    print("** running service down simulation ...")
    print("** learned node embedding from data ...")


def switched_service_simulator():

    normal_trace = {'A': {'B': {'C': None, 'D': {"E": None, "F": None, "G": None}}}}
    switched_service_trace = {'A': {'Q': {'C': None, 'D': {"E": None, "F": None, "G": None}}}}
    one_week_data = [normal_trace]
    new_data = [normal_trace] * 100 + [switched_service_trace]

    print(switched_service_diagram())
    print("=" * 100)
    print("running switched service simulation...")
    print("reading trace data (elapsed one week)...")
    existing_services = set([k for tree in one_week_data for k in tree_utils.tree_to_adjlist(tree).keys()])
    print(f"EXISTING SERVICES: {', '.join(sorted(list(existing_services)))}")
    embeddings = svd_node_embed.svd_node_embed(one_week_data, embed_dim=2)
    print_service_embeddings(embeddings)
    print("learned node embeddings...")
    print("reading more data (elapsed 1 minute)...")

    new_existing_services = set([k for tree in new_data for k in
                             tree_utils.tree_to_adjlist(tree).keys()])
    new_services = new_existing_services.difference(existing_services)
    new_embeddings = svd_node_embed.svd_node_embed(new_data, embed_dim=2)
    print("learned new node embeddings...")
    print_service_embeddings(new_embeddings)
    if new_services:
        for service in new_services:
            print(f"ALERT: FOUND NEW SERVICE {service}")
            print(f"analyzing service {service}...")
            neighbors = nearest_neighbors(new_embeddings, service)
            delta = 0.01
            for neighbor in neighbors:
                distance = neighbor[1]
                neighbor_service = neighbor[0]
                if distance < delta and xor(service, neighbor_service, new_data):
                    print(f"\tdistance({service}, {neighbor_service}) = {distance:0.4f} (< delta of {delta})")
                    print(f"\t ** FLAGGED SERVICE SUBSTITION {neighbor_service} -> {service}")

def print_service_embeddings(embeddings):
    for service in sorted(embeddings.keys()):
        emb = embeddings[service]
        print(f'\t{service} -> [{emb[0]:.2f} {emb[1]:0.2f}]')

def xor(n1, n2, trees):
    adj_lists = [tree_utils.tree_to_adjlist(tree).keys() for tree in trees]
    mutually_exlusive = sum([(n1 in adj_list) != (n2 in adj_list) for adj_list in adj_lists]) == len(trees)
    print(f"\t{n1}, {n2} found mutually exclusive ")
    return mutually_exlusive

def nearest_neighbors(embeddings, node):
    """
    nearest neighbors to the given node, as defined by
    cosine similarity
    """
    node_emb = embeddings[node]
    neighbors = []
    for other_node, other_emb in embeddings.items():
        if other_node != node:
            neighbors.append((other_node, euc_distance(node_emb, other_emb)))
    return sorted(neighbors, key=lambda e: e[1])

def euc_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def service_down_diagram():
    return """
            CASE STUDY: SERVICE DOWN 
            -------------------------
span tree with backend services E1, E2, E3           
occasionally 1% of the time any of E1, E2, E3 go down
        
        A
        |
        B
       / \\
      C   D  
        / | \\
      E1  E2  E3 -> each goes down ~1% of the time                              
    """

def switched_service_diagram():
    return """
                CASE STUDY: REPLACED SERVICE NODE
                -----------------------------------
    
           normal span tree                  service B is seen to be occasionally
                                             switched out for some service Q 1%
                                             of the time

                                                    
                A                                       A
                |                  ====>                |
                B                                       Q (replaces B ~1% of the time)
               / \\                                     / \\
              C   D                                   C   D
                / | \\                                   / | \\
              E1  E2  E3                              E1  E2 E3      
              
              
            ALGORITHM:
            ~~~~~~~~~~ 
            IF DISTANCE(v_B, v_Q) < delta AND XOR(B present, Q present)
                        => FLAG SERVICE_SUBSTITUTION(B, Q)                                                

        """


def main():
    switched_service_simulator()


if __name__ == "__main__":
    main()
