import os
import unittest
from main import run_tests

import tempfile

import numpy as np
import numpy.testing as nptest

import ntimage as nti
import nitrain as nt
from nitrain import transforms as tx

from nitrain.readers.utils import infer_reader

class TestFunction_infer_reader_lists(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_nested_memory_lists(self):
        import ntimage as nti
        from nitrain.readers.utils import infer_reader
        imgs = [nti.example('r16') for _ in range(5)]
        
        reader = infer_reader(imgs)
        self.assertTrue('MemoryReader' in str(type(reader)))
        
        reader = infer_reader([imgs, imgs])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader([[imgs, imgs], imgs])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader([[imgs, imgs], [imgs, imgs]])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('ComposeReader' in str(type(reader.readers[1])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[1])))

    def test_nested_flat_arrays(self):
        import ntimage as nti
        from nitrain.readers.utils import infer_reader
        arr = np.zeros((20,))
        
        reader = infer_reader(arr)
        self.assertTrue('MemoryReader' in str(type(reader)))
        
        reader = infer_reader([arr, arr])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader([[arr, arr], arr])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader([[arr, arr], [arr, arr]])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('ComposeReader' in str(type(reader.readers[1])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[1])))
        
    def test_nested_2d_arrays(self):
        import ntimage as nti
        from nitrain.readers.utils import infer_reader
        arr = np.zeros((20,20))
        
        reader = infer_reader(arr)
        self.assertTrue('MemoryReader' in str(type(reader)))
        
        reader = infer_reader([arr, arr])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader([[arr, arr], arr])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader([[arr, arr], [arr, arr]])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('ComposeReader' in str(type(reader.readers[1])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[1])))


    def test_nested_mixed(self):
        import ntimage as nti
        from nitrain.readers.utils import infer_reader
        arr = np.zeros((20,))
        imgs = [nti.example('r16') for _ in range(5)]
        
        reader = infer_reader(arr)
        self.assertTrue('MemoryReader' in str(type(reader)))
        
        reader = infer_reader([arr, imgs])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader([[arr, imgs], arr])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader([[arr, imgs], [imgs, arr]])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('ComposeReader' in str(type(reader.readers[1])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[1])))

    def test_nested_lists(self):
        import ntimage as nti
        from nitrain.readers.utils import infer_reader
        arr = [0,1,2,3,4]
        
        reader = infer_reader(arr)
        self.assertTrue('MemoryReader' in str(type(reader)))
        
        reader = infer_reader([arr, arr])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader([[arr, arr], arr])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader([[arr, arr], [arr, arr]])
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('ComposeReader' in str(type(reader.readers[1])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[1])))





class TestFunction_infer_reader_dicts(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_nested_memory_lists(self):
        import ntimage as nti
        from nitrain.readers.utils import infer_reader
        imgs = [nti.example('r16') for _ in range(5)]
        
        reader = infer_reader({'x':imgs})
        self.assertTrue('MemoryReader' in str(type(reader)))
        
        reader = infer_reader({'x':imgs, 'y':imgs})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        self.assertEqual(reader.readers[0].label, 'x')
        self.assertEqual(reader.readers[1].label, 'y')
        
        reader = infer_reader({'xy': {'x':imgs, 'y':imgs}, 'z':imgs})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        self.assertEqual(reader.readers[0].label, 'xy')
        self.assertEqual(reader.readers[0].readers[0].label, 'x')
        self.assertEqual(reader.readers[0].readers[1].label, 'y')
        self.assertEqual(reader.readers[1].label, 'z')
        
        reader = infer_reader({'xy': {'x':imgs, 'y':imgs}, 'ab': {'a':imgs, 'b':imgs}})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('ComposeReader' in str(type(reader.readers[1])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[1])))

    def test_nested_flat_arrays(self):
        import ntimage as nti
        from nitrain.readers.utils import infer_reader
        arr = np.zeros((20,))
        
        reader = infer_reader({'x': arr})
        self.assertTrue('MemoryReader' in str(type(reader)))
        
        reader = infer_reader({'x':arr, 'y':arr})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader({'xy': {'x':arr, 'y':arr}, 'z':arr})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader({'xy': {'x':arr, 'y':arr}, 'ab': {'a':arr, 'b':arr}})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('ComposeReader' in str(type(reader.readers[1])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[1])))
        
    def test_nested_2d_arrays(self):
        import ntimage as nti
        from nitrain.readers.utils import infer_reader
        arr = np.zeros((20,20))
        
        reader = infer_reader(arr)
        self.assertTrue('MemoryReader' in str(type(reader)))
        
        reader = infer_reader({'x':arr, 'y':arr})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader({'xy': {'x':arr, 'y':arr}, 'z':arr})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader({'xy': {'x':arr, 'y':arr}, 'ab': {'a':arr, 'b':arr}})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('ComposeReader' in str(type(reader.readers[1])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[1])))


    def test_nested_mixed(self):
        import ntimage as nti
        from nitrain.readers.utils import infer_reader
        arr = np.zeros((20,))
        imgs = [nti.example('r16') for _ in range(5)]
        
        reader = infer_reader(arr)
        self.assertTrue('MemoryReader' in str(type(reader)))
        
        reader = infer_reader({'x':arr, 'y':imgs})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader({'xy': {'x':arr, 'y':imgs}, 'z':arr})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader({'xy': {'x':arr, 'y':imgs}, 'ab': {'a':imgs, 'b':arr}})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('ComposeReader' in str(type(reader.readers[1])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[1])))

    def test_nested_lists(self):
        import ntimage as nti
        from nitrain.readers.utils import infer_reader
        arr = [0,1,2,3,4]
        
        reader = infer_reader(arr)
        self.assertTrue('MemoryReader' in str(type(reader)))
        
        reader = infer_reader({'x':arr, 'y':arr})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader({'xy': {'x':arr, 'y':arr}, 'z':arr})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1])))
        
        reader = infer_reader({'xy': {'x':arr, 'y':arr}, 'ab': {'a':arr, 'b':arr}})
        self.assertTrue('ComposeReader' in str(type(reader)))
        self.assertEqual(len(reader.readers), 2)
        self.assertTrue('ComposeReader' in str(type(reader.readers[0])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[0].readers[1])))
        self.assertTrue('ComposeReader' in str(type(reader.readers[1])))
        self.assertEqual(len(reader.readers[0].readers), 2)
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[0])))
        self.assertTrue('MemoryReader' in str(type(reader.readers[1].readers[1])))
        
        
if __name__ == '__main__':
    run_tests()
