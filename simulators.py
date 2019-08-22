# simulators.py
"""
 data and embedding simulators motivated by
 use cases
"""

def service_down_simulator():
    print(service_down_diagram())
    print("** running service down simulation ...")
    print("** learned node embedding from data ...")


def switched_service_simulator():
    print(switched_service_diagram())
    print("=" * 100)
    print("running switched service simulation...")
    print("reading trace data (elapsed one week)...")
    print("EXISTING SERVICES A, B, C, D, E1, E2, E3")
    print("learned node embeddings...")
    print("reading more data (elapsed 1 minute)...")
    print("ALERT: FOUND NEW SERVICE Q")
    print("learning new node embeddings...")
    print("analyzing service Q...")



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
