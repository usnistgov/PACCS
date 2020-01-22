#!/usr/bin/env python

import copy
import numpy
import sys
import unittest

from paccs import visualization
from paccs import crystal
from paccs import potential
from paccs import minimization
from paccs import wallpaper
from paccs import similarity

_cells = []
for path in ["min_test_1", "min_test_2", "min_test_3", "min_test_4"]:
    with open("data/{}.prim".format(path), "r") as data:
        _cells.append(crystal.CellCodecs.read_cell(data))

class Tests(unittest.TestCase):

    def test_filter_generator(self):
        count = 100
        distance = 3.0
        radii = (2.**(1./6.)/2., 2.**(1./6.)/2.)
        potentials = {
            (0, 0): potential.LennardJonesType(epsilon=0),
            (1, 1): potential.LennardJonesType(epsilon=1),
            (0, 1): potential.LennardJonesType(epsilon=0.5)
            }
        mm = similarity.Histogram(distance, 0.05, 0.99, norm=similarity.Minimum())

        # Generate an ensemble of candidates
        generator = wallpaper.generate_wallpaper((1, 1), place_max=4, random_seed=0,
        sample_count=count, merge_sets=False, sample_groups=[wallpaper.WallpaperGroup(name="p3")])
        results = list(generator)

        # All "count" structures should be provided
        self.assertEqual(len(results), count)

        # Scale cells to "contact"
        scale_cell = lambda cell: crystal.CellTools.scale(cell, cell.vectors / cell.scale_factor(radii))
        cells = [scale_cell(c) for g,c in results]
        energies = numpy.array([potential._evaluate_fast(c, potentials, distance)[0] for c in cells])
        res = zip(cells, energies)

        # Find "unique" ones
        reduced_cells = similarity.reduce(res, mm)
        manually_reduced = len(reduced_cells)
        self.assertEqual(manually_reduced, 89)

        # Do in an automated fashion to compare
        generator2 = wallpaper.generate_wallpaper((1, 1), place_max=4, random_seed=0,
        sample_count=count, merge_sets=False, sample_groups=[wallpaper.WallpaperGroup(name="p3")])
        filter2 = minimization.filter(generator2, potentials, distance, count, radii, similarity_metric=mm)
        self.assertEqual(len(list(filter2)), manually_reduced)

        # Confirm that cells from automatic reduction have not been rescaled
        for idx, (g,c) in enumerate(list(filter2)):
            self.assertTrue(c != reduced_cells[idx][0])
            self.assertTrue(scale_cell(c) == reduced_cells[idx][0])
            self.assertTrue(potential._evaluate_fast(scale_cell(c), potentials, distance)[0]
                == reduced_cells[idx][1])
    
    def test_filter(self):
        generator = wallpaper.generate_wallpaper((1, 1), place_max=5, sample_count=100)
        filter = minimization.filter(generator, {
            (0, 0): potential.LennardJonesType(lambda_=-1),
            (1, 1): potential.LennardJonesType(lambda_=-1),
            (0, 1): potential.LennardJonesType(lambda_=1)},
        4.0, 10)
        self.assertEqual(len(list(filter)), 10)

    def test_filter_radii(self):
        generator = wallpaper.generate_wallpaper((1, 1), place_max=5, sample_count=100)
        filter = minimization.filter(generator, {
            (0, 0): potential.LennardJonesType(lambda_=-1),
            (1, 1): potential.LennardJonesType(lambda_=-1),
            (0, 1): potential.LennardJonesType(lambda_=1)},
        4.0, 10, (0.5, 0.5))
        self.assertEqual(len(list(filter)), 10)

    def test_filter_callback(self):
        generator = wallpaper.generate_wallpaper((1, 1), place_max=5, sample_count=100)
        result = [None]
        def callback(energies):
            result[0] = energies
        filter = minimization.filter(generator, {
            (0, 0): potential.LennardJonesType(lambda_=-1),
            (1, 1): potential.LennardJonesType(lambda_=-1),
            (0, 1): potential.LennardJonesType(lambda_=1)},
        4.0, 10, histogram_callback=callback)

        # Check for proper filtering behavior
        self.assertEqual(len(list(filter)), 10)
        # Get energies (callback should have been triggered)
        energies = result[0]
        sorted_energies = sorted(energies)
        # Check partitioning
        self.assertEqual(energies[10], sorted_energies[10]) # Did partition get target?
        self.assertLessEqual(max(energies[:10]), min(energies[10:])) # Did partition move elements?

    def test_filter_actual_energy(self):
        generator = wallpaper.generate_wallpaper((1, 1), place_max=5, sample_count=100)
        result = [None]
        def callback(energies):
            result[0] = energies
        potentials = {
            (0, 0): potential.LennardJonesType(lambda_=-1),
            (1, 1): potential.LennardJonesType(lambda_=-1),
            (0, 1): potential.LennardJonesType(lambda_=1)}
        distance = 4.0
        filter = minimization.filter(generator, potentials, distance, 20, histogram_callback = callback)

        # Make sure true energies are actually less than cutoff
        filter_results = list(filter) # Exhaust generator to trigger callback
        energies = result[0]
        for cell in filter_results:
            self.assertLessEqual(potential._evaluate_fast(cell[1], potentials, distance)[0], min(energies[20:]))

    def test_decompose_recompose(self):
        for cell in _cells:
            reduced_cell = crystal.CellTools.reduce(cell)
            new_cell = minimization._recompose_cell(reduced_cell, minimization._decompose_cell(reduced_cell))
            self.assertTrue(numpy.all(numpy.isclose(reduced_cell.vectors, new_cell.vectors)))
            for type_index in range(reduced_cell.atom_types):
                self.assertTrue(numpy.all(numpy.isclose(reduced_cell.atoms(type_index), new_cell.atoms(type_index))))

    def test_minimize(self):
        # At the moment, check for no crashes
        run_minimizations(self)

    # Test all sorts of moves: system should never crash
    # There is no good way to actually make sure the moves are behaving as desired from code
    # Suggestion: set _DEBUG_XYZ_PATH, turn off minimizer, and visualize a trajectory for each move type

    def test_exchange_move(self):
        for cell in _cells:
            minimization.optimize(cell, {
                (0, 0): potential.LennardJonesType(lambda_=-1),
                (1, 1): potential.LennardJonesType(lambda_=-1),
                (0, 1): potential.LennardJonesType()}, 6,
                log_level=2, basin_kwargs=dict(niter=10),
                exchange_move=1, vector_move=0, scale_move=0, atom_move=0, cluster_move=0)

    def test_exchange_move_single(self):
        for cell in _cells:
            minimization.optimize(cell, {
                (0, 0): potential.LennardJonesType(lambda_=-1),
                (1, 1): potential.LennardJonesType(lambda_=-1),
                (0, 1): potential.LennardJonesType()}, 6,
                log_level=2, basin_kwargs=dict(niter=10),
                exchange_move=1, vector_move=0, scale_move=0, atom_move=0, cluster_move=0, exchange_select=0.0)

        for cell in _cells:
            minimization.optimize(cell, {
                (0, 0): potential.LennardJonesType(lambda_=-1),
                (1, 1): potential.LennardJonesType(lambda_=-1),
                (0, 1): potential.LennardJonesType()}, 6,
                log_level=2, basin_kwargs=dict(niter=10),
                exchange_move=1, vector_move=0, scale_move=0, atom_move=0, cluster_move=0, exchange_select=-1.0)

    def test_vector_move(self):
        for cell in _cells:
            minimization.optimize(cell, {
                (0, 0): potential.LennardJonesType(lambda_=-1),
                (1, 1): potential.LennardJonesType(lambda_=-1),
                (0, 1): potential.LennardJonesType()}, 6,
                log_level=2, basin_kwargs=dict(niter=10),
                exchange_move=0, vector_move=1, scale_move=0, atom_move=0, cluster_move=0)

    def test_scale_move(self):
        for cell in _cells:
            minimization.optimize(cell, {
                (0, 0): potential.LennardJonesType(lambda_=-1),
                (1, 1): potential.LennardJonesType(lambda_=-1),
                (0, 1): potential.LennardJonesType()}, 6,
                log_level=2, basin_kwargs=dict(niter=10),
                exchange_move=0, vector_move=0, scale_move=1, atom_move=0, cluster_move=0)

    def test_atom_move(self):
        for cell in _cells:
            minimization.optimize(cell, {
                (0, 0): potential.LennardJonesType(lambda_=-1),
                (1, 1): potential.LennardJonesType(lambda_=-1),
                (0, 1): potential.LennardJonesType()}, 6,
                log_level=2, basin_kwargs=dict(niter=10),
                exchange_move=0, vector_move=0, scale_move=0, atom_move=1, cluster_move=0)

    def test_cluster_move(self):
        for cell in _cells:
            minimization.optimize(cell, {
                (0, 0): potential.LennardJonesType(lambda_=-1),
                (1, 1): potential.LennardJonesType(lambda_=-1),
                (0, 1): potential.LennardJonesType()}, 6,
                log_level=2, basin_kwargs=dict(niter=10),
                exchange_move=0, vector_move=0, scale_move=0, atom_move=0, cluster_move=1)

def run_minimizations(test=None):
    for cell in _cells:
        if __name__ == "__main__":
            visualization.cell_mayavi(cell, [3] * cell.dimensions, [1] * cell.atom_types)
        from paccs import similarity
        results = minimization.optimize(cell, {
            (0, 0): potential.LennardJonesType(lambda_=-1, n=12),
            (1, 1): potential.LennardJonesType(lambda_=-1, n=12),
            (0, 1): potential.LennardJonesType(n=12)}, 6,
            log_level=2, initial_kwargs=dict(options=dict(disp=False)),
            # These options are chosen for testing speed, not accurate results!
            basin_kwargs=dict(T=0.3, niter=25, interval=5, niter_success=10, minimizer_kwargs=dict(options=dict(disp=False))),
            final_kwargs=dict(options=dict(disp=False)), save_count=10, save_filter=similarity.Energy(1e-2), save_all=False)
        last_result = results[-1]
        if test:
            test.assertLessEqual(len(results), 10)
            energies = [energy for cell, energy, wall, proc in results]
            for energy_index_1 in range(len(energies)):
                for energy_index_2 in range(energy_index_1 + 1, len(energies)):
                    test.assertGreater(abs(energies[energy_index_1] - energies[energy_index_2]), 1e-2)
            test.assertEqual(last_result, min(results, key=lambda result: result[1]))
        if __name__ == "__main__":
            for index, (cell, energy) in enumerate(results):
                print("{} of {}: {} with E={}".format(index + 1, len(results), cell, energy))
                visualization.cell_mayavi(cell, [3] * cell.dimensions, [1] * cell.atom_types)
