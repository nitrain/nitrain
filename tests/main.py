import sys
import os
import argparse
import unittest
import warnings
import contextlib
from functools import wraps
from itertools import product
from copy import deepcopy

# test that exception occurs:
# with self.assertRaises(Exception):
#       run_fn()

def run_tests():
    unittest.main()