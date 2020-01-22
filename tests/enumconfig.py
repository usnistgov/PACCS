#!/usr/bin/env python

import numpy as np
import unittest
import operator as op

from paccs import enum_config as enumconfig
from functools import reduce

def nCr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, range(n, n-r, -1))
    denom = reduce(op.mul, range(1, r+1))
    return numer//denom

class EnumeratedLatticeTests(unittest.TestCase):

    def setUp(self):
        self.__chunks = [("corners", 4), ("edges", 10), ("faces", 50)]
        self.__recipe = [("corners",{"A":2, "B":1}), ("edges",{"B":1, "D":2}), ("faces",{"A":3, "C":3})]
        self._chunks = {"corners": 3, "edges": 3, "faces": 4}
        self._recipe = {"corners":{"A":1, "B":1}, "edges":{"B":1}, "faces":{"A":2, "B":1}}

    def test_init_bad_chunks(self):
        with self.assertRaises(TypeError):
            ec = enumconfig.EnumeratedLattice(self.__chunks, dict(self.__recipe))

    def test_init_bad_recipe(self):
        with self.assertRaises(TypeError):
            ec = enumconfig.EnumeratedLattice(dict(self.__chunks), self.__recipe)

    def test_bad_name(self):
        chunks = {"corners": 3, "edges": 3, "faces": 4}
        recipe = {"corners":{enumconfig.__EMPTY_SITE__:1, "B":1}, "edges":{"B":1}, "faces":{enumconfig.__EMPTY_SITE__:2, "B":1}}
        with self.assertRaises(ValueError):
            ec = enumconfig.EnumeratedLattice(chunks, recipe)

    def test_init_good(self):
        fail = False
        try:
            enumconfig.EnumeratedLattice(dict(self.__chunks), dict(self.__recipe))
        except Exception as e:
            print(e)
            fail = True
        self.assertFalse(fail)

    def test_integer_names(self):
        chunks = {"corners": 3, "edges": 3, "faces": 4}
        recipe = {"corners":{0:1, 1:1}, "edges":{1:1}, "faces":{0:2, 1:1}}
        fail = False
        try:
            ec = enumconfig.EnumeratedLattice(chunks, recipe)
        except:
            fail = True
        self.assertFalse(fail)

    def test_init_vars(self):
        ec = enumconfig.EnumeratedLattice(dict(self.__chunks), dict(self.__recipe))

        self.assertEqual(ec._EnumeratedLattice__atom_types, ('A', 'B', 'C', 'D'))
        self.assertEqual(ec._EnumeratedLattice__natom_types, 4)
        self.assertEqual(set(dict(self.__chunks).keys()), \
        set(ec._EnumeratedLattice__chunks.keys()))
        self.assertEqual(set(dict(self.__recipe).keys()), \
        set(ec._EnumeratedLattice__recipe.keys()))
        self.assertFalse(ec._EnumeratedLattice__rng)
        self.assertFalse(ec._EnumeratedLattice__used)

    def test_seq(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)

        res = [('corners', [('A', np.array([[0], [1], [2]])),
        ('B', np.array([[0], [1]]))]),
        ('edges', [('B', np.array([[0], [1], [2]]))]),
        ('faces', [('A', np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])), ('B', np.array([[0], [1]]))])
        ]

        a = dict(res)
        for key in a.keys():
            a[key] = dict(a[key])
        b = dict(ec._EnumeratedLattice__sequence)
        for key in b.keys():
            b[key] = dict(b[key])

        self.assertEqual(set([key for key in a.keys()]), set([key for key in b.keys()]))

        for key in a.keys():
            self.assertEqual(set([key2 for key2 in a[key].keys()]), set([key2 for key2 in b[key].keys()]))
            for key2 in a[key].keys():
                self.assertTrue(np.all(a[key][key2] - b[key][key2] < 1.0e-12))

    def test_hyperedge(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)

        self.assertEqual(len(ec._EnumeratedLattice__hyperedge), 5)
        self.assertEqual(ec._EnumeratedLattice__hyperedge[0], nCr(3,1)) # A, corners
        self.assertEqual(ec._EnumeratedLattice__hyperedge[1], nCr(3-1,1)) # B, corners
        self.assertEqual(ec._EnumeratedLattice__hyperedge[2], nCr(3,1)) # B, edges
        self.assertEqual(ec._EnumeratedLattice__hyperedge[3], nCr(4,2)) # A, faces
        self.assertEqual(ec._EnumeratedLattice__hyperedge[4], nCr(4-2,1)) # B, faces

        self.assertEqual(ec._EnumeratedLattice__nconfigs,  nCr(3,1)*nCr(3-1,1)*nCr(3,1)*nCr(4,2)*nCr(4-2,1))

    def test_fill_open1(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)

        x = enumconfig.__EMPTY_SITE__

        test = [x]*5
        ec._fill_open(test, "a", [0,1,4])
        self.assertTrue(np.all(test == ["a", "a", x, x, "a"]))

        ec._fill_open(test, "b", [0])
        self.assertTrue(np.all(test == ["a", "a", "b", x, "a"]))

    def test_fill_open2(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)
        x = enumconfig.__EMPTY_SITE__

        test = [x]*5
        ec._fill_open(test, "a", [2,4])
        self.assertTrue(np.all(test == [x, x, "a", x, "a"]))

        ec._fill_open(test, "b", [1])
        self.assertTrue(np.all(test == [x, "b", "a", x, "a"]))

        ec._fill_open(test, "c", [1])
        self.assertTrue(np.all(test == [x, "b", "a", "c", "a"]))

    def test_fill_open3(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)
        x = enumconfig.__EMPTY_SITE__

        test = [x]*5
        ec._fill_open(test, "a", [0,1,2,3,4])
        self.assertTrue(np.all(test == ["a", "a", "a", "a", "a"]))

        with self.assertRaises(Exception):
            ec._fill_open(test, "b", [0])

    def test_fill_open4(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)
        x = enumconfig.__EMPTY_SITE__

        test = [x]*5
        ec._fill_open(test, "a", [])
        self.assertTrue(np.all(test == [x, x, x, x, x]))

    def test_get_address(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)

        with self.assertRaises(ValueError):
            ec._get_address(ec._EnumeratedLattice__nconfigs + 1)

        # Just spot check a few
        self.assertTrue(np.all(ec._get_address(0) - np.array([0, 0, 0, 0, 0]) < 1.0e-12))
        self.assertTrue(np.all(ec._get_address(1) - np.array([1, 0, 0, 0, 0]) < 1.0e-12))
        self.assertTrue(np.all(ec._get_address(2) - np.array([2, 0, 0, 0, 0]) < 1.0e-12))
        self.assertTrue(np.all(ec._get_address(3) - np.array([0, 1, 0, 0, 0]) < 1.0e-12))
        self.assertTrue(np.all(ec._get_address(4) - np.array([1, 1, 0, 0, 0]) < 1.0e-12))
        self.assertTrue(np.all(ec._get_address(5) - np.array([2, 1, 0, 0, 0]) < 1.0e-12))
        self.assertTrue(np.all(ec._get_address(6) - np.array([0, 0, 1, 0, 0]) < 1.0e-12))

    def test_make_config(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)

        # Spot check a few
        address = [0,0,0,0,0]
        cc = ec._make_config(address)
        res = {"corners":np.array(["A", "B", enumconfig.__EMPTY_SITE__]), \
        "edges":np.array(["B", enumconfig.__EMPTY_SITE__, enumconfig.__EMPTY_SITE__]), \
        "faces":np.array(["A", "A", "B", enumconfig.__EMPTY_SITE__])}
        for key in res.keys():
            self.assertTrue(np.all(res[key] == cc[key]))

        address = [2,1,1,1,0]
        cc = ec._make_config(address)
        res = {"corners":np.array([enumconfig.__EMPTY_SITE__, "B", "A"]), \
        "edges":np.array([enumconfig.__EMPTY_SITE__, "B", enumconfig.__EMPTY_SITE__]), \
        "faces":np.array(["A", "B", "A", enumconfig.__EMPTY_SITE__])}
        for key in res.keys():
            self.assertTrue(np.all(res[key] == cc[key]))

    def test_get_no_repeat(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)

        cc = ec.get(0, repeat=False)
        res = {"corners":np.array(["A", "B", enumconfig.__EMPTY_SITE__]), \
        "edges":np.array(["B", enumconfig.__EMPTY_SITE__, enumconfig.__EMPTY_SITE__]), \
        "faces":np.array(["A", "A", "B", enumconfig.__EMPTY_SITE__])}
        for key in res.keys():
            self.assertTrue(np.all(res[key] == cc[key]))

        with self.assertRaises(Exception):
            ec.get(0, repeat=False)

        cc = ec.get(29, repeat=False)
        res = {"corners":np.array([enumconfig.__EMPTY_SITE__, "B", "A"]), \
        "edges":np.array([enumconfig.__EMPTY_SITE__, "B", enumconfig.__EMPTY_SITE__]), \
        "faces":np.array(["A", "B", "A", enumconfig.__EMPTY_SITE__])}
        for key in res.keys():
            self.assertTrue(np.all(res[key] == cc[key]))

    def test_get_repeat(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)

        cc = ec.get(0, repeat=True)
        res = {"corners":np.array(["A", "B", enumconfig.__EMPTY_SITE__]), \
        "edges":np.array(["B", enumconfig.__EMPTY_SITE__, enumconfig.__EMPTY_SITE__]), \
        "faces":np.array(["A", "A", "B", enumconfig.__EMPTY_SITE__])}
        for key in res.keys():
            self.assertTrue(np.all(res[key] == cc[key]))
        self.assertEqual(ec._EnumeratedLattice__used[0], 1)

        fail = False
        try:
            cc = ec.get(0, repeat=True)
        except:
            fail = True
        self.assertFalse(fail)
        self.assertEqual(ec._EnumeratedLattice__used[0], 2)

    def test_random(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)
        self.assertFalse(ec._EnumeratedLattice__rng)

        fail = False
        try:
            for i in range(0, 50):
                cc = ec.random()
        except Exception as e:
            print(e)
            fail = True
        self.assertFalse(fail)

    def test_all(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)
        ec2 = enumconfig.EnumeratedLattice(self._chunks, self._recipe)
        all_configs = ec.all()

        for idx, c1 in enumerate(all_configs):
            c2 = ec2.get(idx)
            self.assertEqual(set(c1.keys()), set(c2.keys()))
            for k in c1.keys():
                self.assertTrue(np.all(c1[k] == c2[k]))

    def test_all_copy(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)
        all_configs = ec.all()

        for idx, c1 in enumerate(all_configs):
            pass

        self.assertFalse(ec._EnumeratedLattice__used)

    def test_ran(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)
        ran_configs = ec.ran(seed=0)

        used = []
        for idx, c1 in enumerate(ran_configs):
            same = True
            for conf in used:
                for k in c1.keys():
                    if (not np.all(c1[k] == conf[k])):
                        same = False
                        break
            if (len(used) > 0):
                self.assertFalse(same)
            used.append(c1)

        self.assertEqual(idx, 216-1)

    def test_ran_order(self):
        ec = enumconfig.EnumeratedLattice(self._chunks, self._recipe)
        ran_configs1 = next(ec.ran(seed=0))
        ran_configs2 = next(ec.ran(seed=0))
        ran_configs3 = next(ec.ran(seed=1))

        # Just check that first one in each sequence against each other
        # Same seed should produce same results
        same = True
        for k in ran_configs1.keys():
            if (not np.all(ran_configs1[k] == ran_configs2[k])):
                same = False
                break
        self.assertTrue(same)

        # Different seeds, different sequence
        same = True
        for k in ran_configs1.keys():
            if (not np.all(ran_configs1[k] == ran_configs3[k])):
                same = False
                break
        self.assertFalse(same)
