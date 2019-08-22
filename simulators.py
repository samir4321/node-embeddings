# simulators.py
"""
 data and embedding simulators motivated by
 use cases
"""

def service_down_simulator():
    pass


def service_down_diagram():
    return """
    span tree with backend services E1, E2, E3           
    occasionally 1% of the time E1, E2, E3 go down
            
            A
            |
            B
           / \
          C   D  
            / | \
          E1  E2  E3 -> each goes down ~1% of the time                    
          
    """

def switched_service_diagram():
    return """
           normal span tree                  service B is seen to be ocassionally
                                             switched out for some service Q 1%
                                             of the time                                             


                                                    
                A                                       A
                |                  ====>                |
                B                                       Q (replaces B ~1% of the time)
               / \                                     / \
              C   D                                   C   D
                / | \                                   / | \
              E1  E2  E3                              E1  E2 E3

        """