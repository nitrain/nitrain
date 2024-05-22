
import os
import ants
import numpy as np

def get_nitrain_dir():
    if os.environ.get('NITRAIN_DIR') is not None:
        nitrain_dir = os.environ['NITRAIN_DIR']
    else:
        nitrain_dir = os.path.join(os.path.expanduser('~'), '.nitrain')
    
    if not os.path.exists(nitrain_dir):
        os.mkdir(nitrain_dir)
        
    return nitrain_dir    

