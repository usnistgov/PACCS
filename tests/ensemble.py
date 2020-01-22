#!/usr/bin/env python

import fractions
import numpy
import unittest
import multiprocessing

from paccs import ensemble
from paccs import wallpaper
from paccs import potential

_CORES = multiprocessing.cpu_count()

class Tests(unittest.TestCase):

    def test_incongruent_ensemble_count(self):
        ens = ensemble.EnsembleFilter(congruent=False)
        ppot = {
            (0, 0): potential.LennardJonesType(epsilon=1),
            (1, 1): potential.LennardJonesType(epsilon=0.5),
            (0, 1): potential.LennardJonesType(epsilon=0.25)}
        distance = 3.0

        # Allow all of them through since only 64 total
        count = 100 # > 12+52
        generator = ens.filter(['p1', 'p2'], (1,2), 3, ppot, distance, count,
            angles=(numpy.pi/2.0,), length_ratios=(1.0,), cores=_CORES, radii=(0.5,0.5), callback=None)
        self.assertEqual(len([b for a,b in enumerate(generator)]), 12+52)

        # Allow only best 10 through
        count = 10
        generator = ens.filter(['p1', 'p2'], (1,2), 3, ppot, distance, count,
            angles=(numpy.pi/2.0,), length_ratios=(1.0,), cores=_CORES, radii=(0.5,0.5), callback=None)
        self.assertEqual(len([b for a,b in enumerate(generator)]), count)

    def test_incongruent_ensemble_values_all(self):
        ens = ensemble.EnsembleFilter(congruent=False)
        ppot = {
            (0, 0): potential.LennardJonesType(epsilon=1),
            (1, 1): potential.LennardJonesType(epsilon=0.5),
            (0, 1): potential.LennardJonesType(epsilon=0.25)}
        distance = 3.0

        # Sort all the energies if we allow all to go through
        count = 100

        energies = [None, None]
        def filter_callback(filter_energies):
            energies[0] = filter_energies[:count]
            energies[1] = filter_energies[count:]

        generator = ens.filter(['p1', 'p2'], (1,2), 3, ppot, distance, count,
            angles=(numpy.pi/2.0,), length_ratios=(1.0,), cores=_CORES, radii=(0.5,0.5), callback=filter_callback)

        dummy = [b for a,b in enumerate(generator)]
        self.assertEqual(len(energies[0]), 64)
        self.assertEqual(len(energies[1]), 0)
        self.assertTrue(all(energies[0][i] <= energies[0][i+1] for i in range(len(energies[0])-1)))

    def test_energyhistogramcallback(self):
        ens = ensemble.EnsembleFilter(congruent=False)
        ppot = {
            (0, 0): potential.LennardJonesType(epsilon=1),
            (1, 1): potential.LennardJonesType(epsilon=0.5),
            (0, 1): potential.LennardJonesType(epsilon=0.25)}
        distance = 3.0

        # Sort all the energies if we allow all to go through
        count = 10

        cback = ensemble.EnergyHistogramCallback(count=count)

        generator = ens.filter(['p1', 'p2'], (1,2), 3, ppot, distance, count,
            angles=(numpy.pi/2.0,), length_ratios=(1.0,), cores=_CORES, radii=(0.5,0.5), callback=cback)

        dummy = [b for a,b in enumerate(generator)]
        self.assertEqual(len(cback.below_threshold), count)
        self.assertEqual(len(cback.above_threshold), 64-count)
        self.assertTrue(all(cback.below_threshold[i] <= cback.below_threshold[i+1] for i in range(len(cback.below_threshold)-1)))

    def test_incongruent_ensemble_values_fraction(self):
        ens = ensemble.EnsembleFilter(congruent=False)
        ppot = {
            (0, 0): potential.LennardJonesType(epsilon=1),
            (1, 1): potential.LennardJonesType(epsilon=0.5),
            (0, 1): potential.LennardJonesType(epsilon=0.25)}
        distance = 3.0

        # Sort top fraction of energies
        count = 10

        energies = [None, None]
        def filter_callback(filter_energies):
            energies[0] = filter_energies[:count]
            energies[1] = filter_energies[count:]

        generator = ens.filter(['p1', 'p2'], (1,2), 3, ppot, distance, count,
            angles=(numpy.pi/2.0,), length_ratios=(1.0,), cores=_CORES, radii=(0.5,0.5), callback=filter_callback)

        dummy = [b for a,b in enumerate(generator)]
        self.assertEqual(len(energies[0]), count)
        self.assertEqual(len(energies[1]), 64-count)
        self.assertTrue(all(energies[0][i] <= energies[0][i+1] for i in range(len(energies[0])-1)))

    def test_congruent_ensemble_count(self):
        ens = ensemble.EnsembleFilter(congruent=True)
        ppot = {
            (0, 0): potential.LennardJonesType(epsilon=1),
            (1, 1): potential.LennardJonesType(epsilon=0.5),
            (0, 1): potential.LennardJonesType(epsilon=0.25)}
        distance = 3.0

        # Allow all of them through
        count = 2000 # > 1596 which there are for p1 with Ng = 4
        generator = ens.filter(['p1'], (1,2), 4, ppot, distance, count,
            angles=(numpy.pi/2.0,), length_ratios=(1.0,), cores=_CORES, radii=(0.5,0.5), callback=None)
        self.assertEqual(len([b for a,b in enumerate(generator)]), 1596)

        # Allow some of them through
        count = 100 # > 1596 which there are for p1 with Ng = 4
        generator = ens.filter(['p1'], (1,2), 4, ppot, distance, count,
            angles=(numpy.pi/2.0,), length_ratios=(1.0,), cores=_CORES, radii=(0.5,0.5), callback=None)
        self.assertEqual(len([b for a,b in enumerate(generator)]), count)

    def test_congruent_ensemble_values_all(self):
        ens = ensemble.EnsembleFilter(congruent=True)
        ppot = {
            (0, 0): potential.LennardJonesType(epsilon=1),
            (1, 1): potential.LennardJonesType(epsilon=0.5),
            (0, 1): potential.LennardJonesType(epsilon=0.25)}
        distance = 3.0

        # Sort all the energies if we allow all to go through
        count = 2000

        energies = [None, None]
        def filter_callback(filter_energies):
            energies[0] = filter_energies[:count]
            energies[1] = filter_energies[count:]

        generator = ens.filter(['p1'], (1,2), 4, ppot, distance, count,
            angles=(numpy.pi/2.0,), length_ratios=(1.0,), cores=_CORES, radii=(0.5,0.5), callback=filter_callback)

        dummy = [b for a,b in enumerate(generator)]
        self.assertEqual(len(energies[0]), 1596)
        self.assertEqual(len(energies[1]), 0)
        self.assertTrue(all(energies[0][i] <= energies[0][i+1] for i in range(len(energies[0])-1)))

    def test_congruent_ensemble_values_fraction(self):
        ens = ensemble.EnsembleFilter(congruent=False)
        ppot = {
            (0, 0): potential.LennardJonesType(epsilon=1),
            (1, 1): potential.LennardJonesType(epsilon=0.5),
            (0, 1): potential.LennardJonesType(epsilon=0.25)}
        distance = 3.0

        # Sort top fraction of energies
        count = 100

        energies = [None, None]
        def filter_callback(filter_energies):
            energies[0] = filter_energies[:count]
            energies[1] = filter_energies[count:]

        generator = ens.filter(['p1'], (1,2), 4, ppot, distance, count,
            angles=(numpy.pi/2.0,), length_ratios=(1.0,), cores=_CORES, radii=(0.5,0.5), callback=filter_callback)

        dummy = [b for a,b in enumerate(generator)]
        self.assertEqual(len(energies[0]), count)
        self.assertEqual(len(energies[1]), 1596-count)
        self.assertTrue(all(energies[0][i] <= energies[0][i+1] for i in range(len(energies[0])-1)))
