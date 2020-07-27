import unittest
import poly
import numpy as np


class AreaTest():

    def test_area(self):
        p = self.p
        self.assertEqual(poly.integrate00(p['coords']), p['A'])

    

class FullTest(AreaTest):

    def test_i10(self):
        p = self.p
        self.assertEqual(poly.integrate10(p['coords']), p['I10'])

    def test_i01(self):
        p = self.p
        self.assertEqual(poly.integrate01(p['coords']), p['I01'])        

    def test_i20(self):
        p = self.p
        self.assertEqual(poly.integrate20(p['coords']), p['I20'])  

    def test_i02(self):
        p = self.p
        self.assertEqual(poly.integrate02(p['coords']), p['I02'])  

    def test_i11(self):
        p = self.p
        self.assertEqual(poly.integrate11(p['coords']), p['I11'])  



class TestP1(unittest.TestCase, AreaTest):
    # simple square
    p = {
        'coords': np.array([[0,0], [1,0], [1,1], [0,1]]),
        'A': 1.0,
    }


class TestP2(unittest.TestCase, AreaTest):
    # simple triangle
    p = {
        'coords': np.array([[1,1], [2,1], [2,2]]),
        'A': 0.5,
    }


class TestP3(unittest.TestCase, AreaTest):
    # simple triangle, reversed
    p = {
        'coords': np.array([[2,2], [2,1], [1,1]]),
        'A': -0.5,
    }


class TestP4(unittest.TestCase, FullTest):
    # square
    p = {
        'coords': np.array([[-1,-1], [1,-1], [1,1], [-1,1]]),
        'A': 2**2,
        'I10': 0,
        'I01': 0,
        'I20': 2**4/12,
        'I11': 0,
        'I02': 2**4/12,
    }


class TestP5(unittest.TestCase, FullTest):
    # square, not centered
    p = {
        'coords': np.array([[0,0], [2,0], [2,2], [0,2]]),
        'A': 2**2,
        'I10': 4, # A times dist
        'I01': 4,
        'I20': 4 + 2**4/12, # I plus dist**2 x A
        'I11': 4,
        'I02': 4 + 2**4/12,
    }


if __name__ == "__main__":
    unittest.main()