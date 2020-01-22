#!/usr/bin/env python

import numpy
import unittest

from paccs import crystal
from paccs import similarity

class Tests(unittest.TestCase):

    def test_reduce(self):
        test_input = [(None, 1.3), (None, 1.3), (None, 1.299), (None, 0.001),
            (None, -0.001), (None, 0.0009), (None, -2.0), (None, -2.1), (None, -2.1)]
        test_output = [(None, 1.299), (None, 0.0009), (None, -0.001), (None, -2.1),
            (None, -2.0)]
        reduced = similarity.reduce(test_input, similarity.Energy(1.01e-3))

        sorted_reference = sorted(test_output, key=lambda x: x[1])
        sorted_output = sorted(reduced, key=lambda x: x[1])
        self.assertEqual(sorted_reference, sorted_output)

    def test_cache(self):
        # Create a custom metric to keep track of cache hits
        cache_misses = [0]
        class CustomMetric(similarity.SimilarityMetric):
            def _compute_direct(self, cell):
                cache_misses[0] += 1
                return cell[1]
            def __call__(self, cell_1, cell_2):
                return self._compute_cached(cell_1) == self._compute_cached(cell_2)

        # Create a dummy cell to make sure it can get hashed in cache
        cell = crystal.Cell(numpy.eye(3), [numpy.eye(3), numpy.eye(3)])

        # Test cache proper operation
        metric = CustomMetric()
        self.assertFalse(metric((cell, 1), (cell, 2)))
        self.assertFalse(metric((cell, 3), (cell, 4)))
        self.assertEqual(cache_misses[0], 4)
        self.assertFalse(metric((cell, 1), (cell, 4)))
        self.assertEqual(cache_misses[0], 4)
        self.assertFalse(metric((cell, 1), (cell, 5)))
        self.assertEqual(cache_misses[0], 5)
        self.assertTrue(metric((cell, 2), (cell, 2)))
        self.assertTrue(metric((cell, 5), (cell, 5)))
        self.assertEqual(cache_misses[0], 5)

    def test_hybrid(self):
        # Create a custom metric to check calls
        called = [0, 0, 0]
        class CustomMetric(similarity.SimilarityMetric):
            def __init__(self, index, offset):
                self.__index = index
                self.__offset = offset
            def __call__(self, cell_1, cell_2):
                called[self.__index] += 1
                return cell_1[1] == cell_2[1] + self.__offset

        # Hybrid OR: 1st true -> don't evaluate any more
        self.assertTrue(similarity.Hybrid(True, CustomMetric(0, 0),
            CustomMetric(1, 1), CustomMetric(2, 1))((None, 1), (None, 1)))
        self.assertEqual(called, [1, 0, 0])

        # Hybrid OR: go through until true is found
        self.assertTrue(similarity.Hybrid(True, CustomMetric(0, 0),
            CustomMetric(1, 1), CustomMetric(2, 1))((None, 1), (None, 0)))
        self.assertEqual(called, [2, 1, 0])

        # Hybrid AND: stop as soon as something is not true
        self.assertFalse(similarity.Hybrid(False, CustomMetric(0, 0),
            CustomMetric(1, 1), CustomMetric(2, 1))((None, 0), (None, 0)))
        self.assertEqual(called, [3, 2, 0])

        # Hybrid AND: evaluate ALL to check for truth
        self.assertTrue(similarity.Hybrid(False, CustomMetric(0, -4),
            CustomMetric(1, -4), CustomMetric(2, -4))((None, 0), (None, 4)))
        self.assertEqual(called, [4, 3, 1])

    def test_energy(self):
        self.assertTrue(similarity.Energy(1e-3)((None, 1.3), (None, 1.3005)))
        self.assertTrue(similarity.Energy(1e-1)((None, 0.05), (None, -0.05)))
        self.assertFalse(similarity.Energy(1e-2)((None, 1.4), (None, 1.42)))
        self.assertFalse(similarity.Energy(1e-10)((None, 0), (None, 1e-9)))

    def test_histogram(self):
        with open("data/min_test_1.prim") as f: nacl = crystal.CellCodecs.read_cell(f)
        with open("data/min_test_2.prim") as f: zns = crystal.CellCodecs.read_cell(f)

        # Identical cells should be identical
        self.assertTrue(similarity.Histogram(6, 0.1, 1)((nacl, None), (nacl, None)))
        # Similarity metric indicates that differing cells are truly different
        self.assertFalse(similarity.Histogram(6, 0.1, 0.9)((nacl, None), (zns, None)))
        # Eventually, this will fail
        self.assertTrue(similarity.Histogram(6, 0.1, 0.3)((nacl, None), (zns, None)))

    def test_histogram_scale(self):
        # Make a sample AB and AB2 cell
        cell1 = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), 0.5 * numpy.ones((1, 3))])
        cell2 = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), \
            numpy.array([[0.35, 0.35, 0.35], [0.75, 0.75, 0.75]])])

        # Check similarity metric
        self.assertTrue(similarity.Histogram(6, 0.1, 0.999)((cell1, None), (cell1, None)))
        self.assertTrue(similarity.Histogram(6, 0.1, 0.999)((cell1, None), (crystal.CellTools.tile(cell1, (3, 2, 4)), None)))
        self.assertFalse(similarity.Histogram(6, 0.1, 0.999)((cell1, None), (cell2, None)))

        # Check numeric scaling
        histogram1 = similarity.Histogram(6, 0.1, 0.999)._compute_direct(cell1)
        histogram1s = similarity.Histogram(6, 0.1, 0.999)._compute_direct(crystal.CellTools.tile(cell1, (3, 2, 4)))
        histogram2 = similarity.Histogram(6, 0.1, 0.999)._compute_direct(cell2)
        self.assertTrue(numpy.all(numpy.isclose(histogram1, histogram1s)))
        self.assertTrue(numpy.all(numpy.isclose(histogram1[0, 1], histogram1[1, 0])))
        self.assertFalse(numpy.all(numpy.isclose(histogram2[0, 1], histogram2[1, 0])))
