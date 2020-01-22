#!/usr/bin/env python

import fractions
import numpy
import unittest

from paccs import crystal
from paccs import wallpaper
from paccs import enum_config as enumconfig

class Tests(unittest.TestCase):
    def test_group_definitions(self):
        # Check wallpaper group definition data for consistency
        for group in wallpaper._wallpaper_groups:
            self.assertIsInstance(group, wallpaper._WallpaperGroup)
            self.assertIsInstance(group.number, int)
            self.assertIsInstance(group.name, str)
            self.assertIsInstance(group.symbol, str)
            self.assertIsInstance(group.half, bool)
            self.assertIsInstance(group.dot, (float, type(None)))
            self.assertIsInstance(group.ratio, (float, type(None)))
            self.assertIsInstance(group.corners, list)
            for corner_symmetry in group.corners:
                self.assertIsInstance(corner_symmetry, set)
                self.assertGreaterEqual(len(corner_symmetry), 1)
                for corner in corner_symmetry:
                    self.assertIsInstance(corner, int)
            self.assertIsInstance(group.edges, list)
            for edge_symmetry in group.edges:
                self.assertIsInstance(edge_symmetry, wallpaper._EdgeSymmetry)
                self.assertGreaterEqual(len(edge_symmetry), 1)
                for edge in edge_symmetry:
                    self.assertIsInstance(edge, int)
            self.assertIsInstance(group.vectors, list)
            self.assertEqual(len(group.vectors), 2)
            for vector in group.vectors:
                self.assertIsInstance(vector, tuple)
                self.assertEqual(len(vector), 2)
                for vector_component in vector:
                    self.assertIsInstance(vector_component, float)
            self.assertIsInstance(group.copies, list)
            self.assertGreaterEqual(len(group.copies), 1)
            for copy in group.copies:
                self.assertIsInstance(copy, list)
                for command in copy:
                    self.assertIsInstance(command, wallpaper._PeriodicOperation)
                    self.assertEqual(len(command), 1 if isinstance(command, wallpaper._ROT) else 2)
                    for value in command:
                        self.assertIsInstance(value, int if isinstance(command, wallpaper._REF) else float)
            self.assertIsInstance(group.stoichiometry, list)
            self.assertEqual(len(group.stoichiometry), 3 if group.half else 4)
            for value in group.stoichiometry:
                self.assertIsInstance(value, fractions.Fraction)
                self.assertLess(value, fractions.Fraction(1, 2))

    def test_wallpaper_group(self):
        # Make sure groups can be selected by number, name, or symbol
        for group in wallpaper._wallpaper_groups:
            self.assertIs(group, wallpaper.WallpaperGroup(number=group.number))
            self.assertIs(group, wallpaper.WallpaperGroup(name=group.name))
            self.assertIs(group, wallpaper.WallpaperGroup(symbol=group.symbol))

    def test_tile_wallpaper(self):
        # Create a chiral 2D cell
        reference_cell_1 = crystal.Cell(numpy.eye(2), \
            [numpy.array([[0.25, 0.25]]), numpy.array([[0.75, 0.25]]), numpy.array([[0.25, 0.75]])], ["O", "X", "Y"])
        a, b = 0.18469903125906464, 0.554097093777194 # to get nice contacts for right-angled 45-45-90 triangles
        reference_cell_2 = crystal.Cell(numpy.eye(2), \
            [numpy.array([[a, a]]), numpy.array([[b, a]]), numpy.array([[a, b]])], ["O", "X", "Y"])

        # Create the tilings (for possible visual inspection later)
        for group_index in range(17):
            group = wallpaper.WallpaperGroup(number=group_index + 1)

            # Determine what the cell vectors should look like
            length_ratio = group.ratio if group.ratio is not None else 1.25
            unit_dot = group.dot if group.dot is not None else 0.25
            vectors = numpy.array([[length_ratio, 0.0], [unit_dot, numpy.sqrt(1 - (unit_dot ** 2))]])

            # Do it
            tiling_cell = crystal.CellTools.scale(reference_cell_2 if group.half else reference_cell_1, vectors)
            tiled_cell = wallpaper.tile_wallpaper(tiling_cell, group)
            with open("data/group_{}.cell".format(group_index + 1), "w") as data_file:
                crystal.CellCodecs.write_cell(tiled_cell, data_file)

    def test_wallpaper_stoichiometry(self):
        # Check all stoichiometries against reference data
        # If this test fails, something changed in the module setup routine
        # Normalized such that lowest value is 1
        reference_stoichiometries = [
            # Corners       Edge offs.      Edge ctrs.      Faces
            [1, 1, 1, 1,    1, 1, 1, 1,     1, 1, 1, 1,     1],
            [1, 1, 1, 1,    2, 2, 2, 2,     1, 1, 2, 2,     2],
            [1, 1, 1, 1,    2, 2, 1, 1,     2, 2, 1, 1,     2],
            [1, 1, 1, 1,    1, 1, 1, 1,     1, 1, 1, 1,     1],
            [1, 1, 1, 1,    2, 2, 2, 2,     2, 2, 2, 2,     4],
            [1, 1, 1, 1,    1, 1, 2, 2,     1, 1, 1, 1,     2],
            [1, 1, 1, 1,    2, 2, 2, 2,     2, 2, 2, 2,     2],
            [1, 1, 1, 1,    1, 1, 2, 2,     1, 1, 2, 2,     2],
            [1, 1, 1,       2, 2, 4,        2, 2, 2,        4],
            [1, 2, 2, 1,    4, 4, 4, 4,     4, 4, 4, 4,     4],
            [2, 1, 1,       4, 4, 4,        4, 4, 4,        8],
            [1, 1, 1,       4, 4, 2,        4, 4, 2,        4],
            [1, 1, 1, 1,    3, 3, 3, 3,     3, 3, 3, 3,     3],
            [1, 1, 1,       3, 3, 3,        3, 3, 3,        6],
            [2, 1, 1,       6, 6, 3,        6, 6, 3,        6],
            [2, 1, 1,       6, 6, 6,        6, 6, 3,        6],
            [2, 1, 3,       6, 6, 6,        6, 6, 6,        12]]
        self.assertEqual(wallpaper._wallpaper_stoichiometries, reference_stoichiometries)

    def _test_generator_wallpaper_overall(self, **kwargs):
        results = list(wallpaper.generate_wallpaper(sample_count=5, log_level=1, **kwargs))
        for group, result in results:
            # Check stoichiometry
            expected_stoichiometry = numpy.array(kwargs["stoichiometry"], dtype=float)
            actual_stoichiometry = numpy.array(result.atom_counts, dtype=float)
            expected_stoichiometry /= numpy.sum(expected_stoichiometry)
            actual_stoichiometry /= numpy.sum(actual_stoichiometry)
            self.assertTrue(numpy.all(numpy.isclose(expected_stoichiometry, actual_stoichiometry)))

    def test_generate_wallpaper_overall(self):
        stoichiometries = [(1, 1), (2, 1), (2, 2), (5, 4)]
        places = [(1, None), (3, None), (2, 10), (6, 6)]
        grids = [3, 4, 5]
        sample_selection = [(True, True, True, True), (True, False, False, False), \
            (False, True, True, False), (False, True, True, True)]
        sample_groups_options = [None, [wallpaper.WallpaperGroup(name="p4")], \
            [wallpaper.WallpaperGroup(name="p1"), wallpaper.WallpaperGroup(name="cm"), \
            wallpaper.WallpaperGroup(name="p3m1")]]

        for stoichiometry in stoichiometries:
            # Test different place options
            for place_min, place_max in places:
                self._test_generator_wallpaper_overall(stoichiometry=stoichiometry, \
                    place_min=place_min, place_max=place_max, grid_count=3)

            # Test different grid options
            for grid_count in grids:
                self._test_generator_wallpaper_overall(stoichiometry=stoichiometry, \
                    grid_count=grid_count, place_min=1, place_max=6)

            # Test different selection options
            for sample_corners, sample_edge_offsets, sample_edge_centers, sample_faces in sample_selection:
                self._test_generator_wallpaper_overall(stoichiometry=stoichiometry, sample_corners=sample_corners, \
                    sample_edge_offsets=sample_edge_offsets, sample_edge_centers=sample_edge_centers, sample_faces=sample_faces, \
                    grid_count=3, place_min=1, place_max=6)

            # Test different group options
            for sample_groups in sample_groups_options:
                self._test_generator_wallpaper_overall(stoichiometry=stoichiometry, \
                    grid_count=3, sample_groups=sample_groups, place_min=1, place_max=6)

    def test_generate_wallpaper_interpolation(self):
        stoichiometry = (2, 3)
        angle_tests = [(numpy.pi / 8, numpy.pi / 3), (1, 1.5, 2)]
        length_tests = [(0.5, 1.0, 1.5), (0.7, 0.9, 2.0)]

        # Test different angle options
        for angles in angle_tests:
            self._test_generator_wallpaper_overall(stoichiometry=stoichiometry, place_min=1, place_max=6, \
                angles=angles)

        # Test different length options
        for lengths in length_tests:
            self._test_generator_wallpaper_overall(stoichiometry=stoichiometry, place_min=1, place_max=6, \
                length_ratios=lengths)

    def test_uniqueness_count_single(self):
        groups = [wallpaper.WallpaperGroup(name="p1")]

        gen = wallpaper.generate_wallpaper(stoichiometry=(1,2), length_ratios=(1,), \
            angles=(numpy.pi/2,), grid_count=2, sample_groups=groups)
        self.assertEqual(len([c for c in enumerate(gen)]), 0)

        gen = wallpaper.generate_wallpaper(stoichiometry=(1,2), length_ratios=(1,), \
            angles=(numpy.pi/2,), grid_count=3, sample_groups=groups)
        self.assertEqual(len([c for c in enumerate(gen)]), 12)

        gen = wallpaper.generate_wallpaper(stoichiometry=(1,2), length_ratios=(1,), \
            angles=(numpy.pi/2,), grid_count=4, sample_groups=groups)
        self.assertEqual(len([c for c in enumerate(gen)]), 1596)

    def test_uniqueness_count_single_congruent(self):
        # p1 should be unaffected
        groups = [wallpaper.WallpaperGroup(name="p1")]
        gen = wallpaper.generate_wallpaper(stoichiometry=(1,2), length_ratios=(1,), \
            angles=(numpy.pi/2,), grid_count=3, sample_groups=groups, log_level=0,
            congruent=True)
        self.assertEqual(len([c for c in enumerate(gen)]), 12)

        # p2 with lr = 1 should have same as Ng = 3 if congruent to Ng = 5 (3 = floor(5/sqrt(2*1)))
        groups = [wallpaper.WallpaperGroup(name="p2")]
        gen = wallpaper.generate_wallpaper(stoichiometry=(1,2), length_ratios=(1,), \
            angles=(numpy.pi/2,), grid_count=5, sample_groups=groups, log_level=0,
            congruent=True)
        self.assertEqual(len([c for c in enumerate(gen)]), 52)

        # p6m with lr = 1 should have same as Ng = 8 if congruent to Ng = 3 (3 = floor(8/sqrt(6)))
        groups = [wallpaper.WallpaperGroup(name="p6m")]
        gen = wallpaper.generate_wallpaper(stoichiometry=(1,2), length_ratios=(1,), \
            angles=(numpy.pi/2,), grid_count=8, sample_groups=groups, log_level=0,
            congruent=True)
        self.assertEqual(len([c for c in enumerate(gen)]), 29)

    def test_uniqueness_count_multiple(self):
        groups = [wallpaper.WallpaperGroup(name="p1")]
        gen = wallpaper.generate_wallpaper(stoichiometry=(1,2), length_ratios=(1,2,), \
            angles=(numpy.pi/2,), grid_count=3, sample_groups=groups)
        self.assertEqual(len([c for c in enumerate(gen)]), 12 + 588)

        groups = [wallpaper.WallpaperGroup(name="p1"), wallpaper.WallpaperGroup(name="p2")]
        gen = wallpaper.generate_wallpaper(stoichiometry=(1,2), length_ratios=(1,), \
            angles=(numpy.pi/2,), grid_count=3, sample_groups=groups)
        self.assertEqual(len([c for c in enumerate(gen)]), 12+52)

    def test_uniqueness_set(self):
        groups = [wallpaper.WallpaperGroup(name="p1")]
        gen = wallpaper.generate_wallpaper(stoichiometry=(1,2), length_ratios=(1,), \
            angles=(numpy.pi/2,), grid_count=3, sample_groups=groups, debug=True)

        res = set()
        for (idx, (g, struct)) in enumerate(gen):
            self.assertTrue(struct not in res)
            res.add(struct)
        self.assertEqual(len(res), 12)

        groups = [wallpaper.WallpaperGroup(name="p2")]
        gen = wallpaper.generate_wallpaper(stoichiometry=(1,2), length_ratios=(1,), \
            angles=(numpy.pi/2,), grid_count=3, sample_groups=groups, debug=True)

        res = set()
        for (idx, (g, struct)) in enumerate(gen):
            self.assertTrue(struct not in res)
            res.add(struct)
        self.assertEqual(len(res), 52)

    def test_minimum_configurations_single_group(self):
        gen = wallpaper.generate_wallpaper(stoichiometry=(1,1,1), length_ratios=(1.,), \
            angles=(numpy.pi/2,), grid_count=3, sample_groups=[wallpaper.WallpaperGroup(name='p2')],
            congruent=False,log_level=0,
            minimum_configurations=96)
        self.assertEqual(len([c for c in enumerate(gen)]), 96)

        gen = wallpaper.generate_wallpaper(stoichiometry=(1,1,1), length_ratios=(1.,), \
            angles=(numpy.pi/2,), grid_count=3, sample_groups=[wallpaper.WallpaperGroup(name='p2')],
            congruent=False,log_level=0,
            minimum_configurations=97)
        self.assertEqual(len([c for c in enumerate(gen)]), 9744)

    def test_minimum_configurations_multiple_groups(self):
        gen = wallpaper.generate_wallpaper(stoichiometry=(1,1,1), length_ratios=(1.,), \
            angles=(numpy.pi/2,), grid_count=3, sample_groups=[wallpaper.WallpaperGroup(name='p3'),
                                                           wallpaper.WallpaperGroup(name='p2'),],
            congruent=False,log_level=0,
                minimum_configurations=162)
        self.assertEqual(len([c for c in enumerate(gen)]), 162)

        gen = wallpaper.generate_wallpaper(stoichiometry=(1,1,1), length_ratios=(1.,), \
            angles=(numpy.pi/2,), grid_count=3, sample_groups=[wallpaper.WallpaperGroup(name='p3'),
                                                           wallpaper.WallpaperGroup(name='p2'),],
            congruent=False,log_level=0,
                minimum_configurations=163)
        self.assertEqual(len([c for c in enumerate(gen)]), 36630)
