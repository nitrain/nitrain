import os
import unittest

from tempfile import mkdtemp
import shutil
import pandas as pd

import numpy as np
import ants
import nitrain as nt
from nitrain import readers, transforms as tx
        
from main import run_tests

class TestClass_Bugs(unittest.TestCase):
    def setUp(self):
        pass
         
    def tearDown(self):
        pass
    
if __name__ == '__main__':
    run_tests()
