import sys
import os
import argparse
import unittest
import warnings
import contextlib
from functools import wraps
from itertools import product
from copy import deepcopy

def run_tests():
    unittest.main()