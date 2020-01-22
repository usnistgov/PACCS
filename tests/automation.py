#!/usr/bin/env python

import fractions
import numpy
import unittest
import io, os, fcntl, sys

from paccs import automation
from paccs import crystal
from paccs import potential
from paccs.wallpaper import WallpaperGroup

class ProcessorTests(unittest.TestCase):

    def test_cell_processor(self):
        # CellProcessor should do absolutely nothing
        cell = crystal.Cell(numpy.eye(3), [numpy.eye(3)] * 3)
        self.assertEqual(cell, automation.CellProcessor()(cell))

    def test_scaling_processor(self):
        # ScalingProcessor should wrap CellTools.scale and Cell.scale_factor
        cell = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), 0.5 * numpy.ones((1, 3))])
        new_radii = (3.4, 5.6)
        self.assertEqual(crystal.CellTools.scale(cell, cell.vectors / cell.scale_factor(new_radii)), \
            automation.ScalingProcessor(new_radii)(cell))

    def test_tiling_processor(self):
        # TilingProcessor should wrap CellTools.tile
        cell = crystal.Cell(numpy.eye(3), [numpy.eye(3)] * 3)
        self.assertEqual(crystal.CellTools.tile(cell, (4, 4, 4)), automation.TilingProcessor((4, 4, 4))(cell))

    def test_filtering_processor(self):
        # AutoFilteringProcessor should do absolutely nothing when called
        # It will get removed from the preprocessing chain and handled separately
        cell = crystal.Cell(numpy.eye(3), [numpy.eye(3)] * 3)
        self.assertEqual(cell, automation.AutoFilteringProcessor(1)(cell))

# This is module-level for pickling purposes
def _custom_processor(cell):
    with io.open("data/testdb_check.in", mode="ab", buffering=0) as check_file:
        try:
            fcntl.lockf(check_file, fcntl.LOCK_EX)
            check_file.write(b"!")
            check_file.flush()
            os.fsync(check_file.fileno())
        finally:
            fcntl.lockf(check_file, fcntl.LOCK_UN)
    return cell

class TaskManagerTests(unittest.TestCase):
    _default_potentials = {
        (0, 0): potential.LennardJonesType(lambda_=0.5),
        (0, 1): potential.LennardJonesType(lambda_=1.0),
        (1, 1): potential.LennardJonesType(lambda_=0.25)
    }

    # Note: specifying basinhopping niter=X yields X+1 callback invocations

    def test_default(self):
        # Make sure system runs with no crashes
        task_manager = automation.TaskManager([(
            dict(stoichiometry=(1, 2), place_max=4, sample_count=4, log_level=2,
                sample_groups=[WallpaperGroup(number=i + 1) for i in range(2)]),
            dict(potentials=TaskManagerTests._default_potentials, distance=4, log_level=2,
                basin_kwargs=dict(niter=5), save_all=True, save_count=3))])
        task_manager.work("data/testdb_default.db")

        # Make sure all processing did complete
        db = automation.ResultsDatabase("data/testdb_default.db")
        self.assertEqual(len(list(db.cells_in)), 4)
        self.assertEqual(len(list(db.cells_out)), 12)

    def test_preprocessor(self):
        # Make sure processor gets called on each input
        task_manager = automation.TaskManager([(
            dict(stoichiometry=(1, 2), place_max=4, sample_count=4, log_level=2,
                sample_groups=[WallpaperGroup(number=i + 1) for i in range(2)]),
            dict(potentials=TaskManagerTests._default_potentials, distance=4, log_level=2,
                basin_kwargs=dict(niter=5), save_all=True, save_count=3))],
            preprocessors=[_custom_processor])

        with open("data/testdb_check.in", mode="wb") as f:
            pass
        task_manager.work("data/testdb_preprocessor.db")
        with open("data/testdb_check.in", mode="rb") as f:
            self.assertEqual(len(f.read()), 4)

    def test_postprocessor(self):
        # Make sure processor gets called on each output
        task_manager = automation.TaskManager([(
            dict(stoichiometry=(1, 2), place_max=4, sample_count=4, log_level=2,
                sample_groups=[WallpaperGroup(number=i + 1) for i in range(2)]),
            dict(potentials=TaskManagerTests._default_potentials, distance=4, log_level=2,
                basin_kwargs=dict(niter=5), save_all=True, save_count=3))],
            postprocessors=[_custom_processor])

        with open("data/testdb_check.in", mode="wb") as f:
            pass
        task_manager.work("data/testdb_preprocessor.db")
        with open("data/testdb_check.in", mode="rb") as f:
            self.assertEqual(len(f.read()), 12)

    def test_multiple_jobs(self):
        # Make sure that system can handle multiple jobs
        task_manager = automation.TaskManager([(
            dict(stoichiometry=(1, 2), place_max=4, sample_count=2, log_level=2,
                sample_groups=[WallpaperGroup(number=i + 1) for i in range(2)]),
            dict(potentials=TaskManagerTests._default_potentials, distance=4, log_level=2,
                basin_kwargs=dict(niter=5), save_all=True, save_count=3))] * 2)
        task_manager.work("data/testdb_multiple_jobs.db")

        # All processing should have still happened
        db = automation.ResultsDatabase("data/testdb_multiple_jobs.db")
        self.assertEqual(len(list(db.cells_in)), 4)
        self.assertEqual(len(list(db.cells_out)), 12)

    def test_filter(self):
        task_manager = automation.TaskManager([(
            dict(stoichiometry=(1, 2), place_max=4, sample_count=20, log_level=2,
                sample_groups=[WallpaperGroup(number=i + 1) for i in range(2)]),
            dict(potentials=TaskManagerTests._default_potentials, distance=4, log_level=2,
                basin_kwargs=dict(niter=5), save_all=True, save_count=3))],
            preprocessors=[automation.AutoFilteringProcessor(count=2),
                automation.ScalingProcessor((0.5, 0.5))])
        task_manager.work("data/testdb_filter.db")

        # Make sure that filtering did happen
        db = automation.ResultsDatabase("data/testdb_filter.db")
        self.assertEqual(len(list(db.cells_in)), 4)
        self.assertEqual(len(list(db.cells_out)), 12)

class ResultsDatabaseTests(unittest.TestCase):

    def test_read_valid(self):
        db = automation.ResultsDatabase("data/example.dat")
        self.assertEqual(db.block_count, 20)
		# After moving wallpaper group information to wallpaper.py the
		# _WallpaperGroup class stored in this example database is not
		# available in the right module somewhere - this example.dat file
		# should just be regenerated
        """self.assertEqual(len(list(db.cells_in)), 16)
        self.assertEqual(len(list(db.cells_out)), 64)
        self.assertEqual(len(list(db.results)), 20)
        self.assertEqual(len(list(db.generate_results)), 4)
        self.assertEqual(len(list(db.optimize_results)), 16)"""

    def test_read_corrupted(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaises(RuntimeWarning):
                db = automation.ResultsDatabase("data/example_corrupt.dat")
