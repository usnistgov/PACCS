#!/usr/bin/env python

import io
import numpy
import unittest

from paccs import crystal

_test_good_vectors = [
    numpy.eye(2),
    numpy.array([[1, 0], [0.5, 1.1]]),
    numpy.array([[1, 1], [-1, 1]]),
    numpy.array([[0, -1], [-1, 0]]),
    numpy.eye(3),
    numpy.diag(numpy.arange(1, 4)),
    numpy.array([[1, 2, 3], [6, 4, 5], [8, 9, 7]]),
    numpy.array([[1, -2, 3], [-6, 4, -5], [8, -9, 7]]),
    numpy.array([[-1, -2, -3], [6, 4, 5], [8, 9, 7]])]

class CellTests(unittest.TestCase):

    def test_init_bad_vectors(self):
        # Reject bad input vectors
        test_vectors = [
            numpy.eye(1),
            numpy.eye(2, 3)]
        for vectors in test_vectors:
            with self.assertRaises(Exception):
                crystal.Cell(vectors, [])

    def test_init_good_vectors(self):
        # Accept good input vectors
        test_vectors = _test_good_vectors
        for vectors in test_vectors:
            cell = crystal.Cell(vectors, [])
            self.assertTrue(numpy.all(vectors == cell.vectors))

    def test_init_bad_atom_lists(self):
        # Reject bad atom lists
        test_lists = [
            [numpy.eye(2)],
            [numpy.arange(1, 10)],
            [numpy.eye(3), numpy.eye(2)]]
        for lists in test_lists:
            with self.assertRaises(Exception):
                crystal.Cell(numpy.eye(3), lists)

    def test_init_good_atom_lists(self):
        # Accept good atom lists
        test_lists = [
            [numpy.eye(3)],
            [numpy.ones((2, 3)), numpy.ones((4, 3))],
            [numpy.eye(3), numpy.ones((2, 3)), numpy.zeros((4, 3))]]
        for lists in test_lists:
            cell = crystal.Cell(numpy.eye(3), lists)
            self.assertTrue(all(numpy.all(lists[index] == cell.atoms(index)) for index in range(len(lists))))

    def test_init_bad_names(self):
        # Reject bad atom names
        test_names = [
            ["H"],
            ["H", "H"],
            ["H", "He", "Li"]]
        for names in test_names:
            with self.assertRaises(Exception):
                crystal.Cell(numpy.eye(3), [numpy.eye(3)] * 2, names)

    def test_init_good_names(self):
        # Accept good atom names
        test_names = [
            ["H", "He"],
            ["A", "B"],
            None]
        for names in test_names:
            cell = crystal.Cell(numpy.eye(3), [numpy.eye(3)] * 2, names)
            if names is not None:
                self.assertTrue(names == cell.names)

    def test_equality(self):
        reference = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), 0.5 * numpy.ones((2, 3))])
        identical = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), 0.5 * numpy.ones((2, 3))])
        near_identical = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)) + 1e-8, 0.5 * numpy.ones((2, 3))])
        different_numbers = crystal.Cell(numpy.eye(3), [numpy.ones((1, 3)), numpy.zeros((2, 3))])
        different_shapes = crystal.Cell(numpy.eye(2), [numpy.zeros((1, 2)), 0.5 * numpy.ones((3, 2))])

        def test_equality_case(cell_1, cell_2, expected_result):
            self.assertEqual(cell_1 == cell_2, expected_result)
            self.assertNotEqual(cell_1 != cell_2, expected_result)
            self.assertEqual(hash(cell_1) == hash(cell_2), expected_result)

        test_equality_case(reference, reference, True)
        test_equality_case(reference, identical, True)
        test_equality_case(reference, near_identical, False)
        test_equality_case(reference, different_numbers, False)
        test_equality_case(reference, different_shapes, False)

    def test_dimensions(self):
        # Check proper return of dimensions
        for dimensions in range(2, 5):
            self.assertEqual(crystal.Cell(numpy.eye(dimensions), []).dimensions, dimensions)

    def test_vectors(self):
        # Check proper return of vectors
        for vectors in _test_good_vectors:
            self.assertTrue(numpy.all(crystal.Cell(vectors, []).vectors == vectors))

    def test_vector(self):
        # Check proper return of individual vectors
        for vectors in _test_good_vectors:
            for index in range(vectors.shape[0]):
                self.assertTrue(numpy.all(crystal.Cell(vectors, []).vector(index) == vectors[index]))

    def test_enclosed(self):
        # Check proper return of enclosed space
        for vectors in _test_good_vectors:
            self.assertAlmostEqual(numpy.abs(numpy.linalg.det(vectors)), crystal.Cell(vectors, []).enclosed)

    def test_surface(self):
        # Check proper return of surface space
        for vectors in _test_good_vectors:
            if vectors.shape[0] == 2:
                # Manually calculate perimeter
                surface = 2 * sum(numpy.linalg.norm(vectors[i]) for i in range(vectors.shape[0]))
            else:
                # Manually calculate surface area
                surface = 2 * sum(numpy.linalg.norm(numpy.cross(vectors[i - 1], \
                    vectors[i])) for i in range(vectors.shape[0]))
            self.assertAlmostEqual(surface, crystal.Cell(vectors, []).surface)

    def test_distortion_factor(self):
        # Check proper calculation of distortion factor
        for vectors in _test_good_vectors:
            # Should be 1 for squares and cubes, greater for all others
            normalized = crystal.CellTools.normalize(crystal.Cell(vectors, [])).vectors
            if numpy.all(numpy.isclose(normalized / normalized[0, 0], numpy.eye(vectors.shape[0]))):
                self.assertAlmostEqual(crystal.Cell(vectors, []).distortion_factor, 1)
            else:
                self.assertGreater(crystal.Cell(vectors, []).distortion_factor, 1)

    def test_normals(self):
        # Check proper calculation of normals
        for vectors in _test_good_vectors:
            normals = crystal.Cell(vectors, []).normals
            for i in range(vectors.shape[0]):
                if vectors.shape[0] == 2:
                    self.assertAlmostEqual(numpy.dot(normals[i - 1], vectors[i]), 0)
                else:
                    self.assertTrue(numpy.all(numpy.isclose(normals[i], \
                        numpy.cross(vectors[(i + 1) % 3], vectors[(i + 2) % 3]))))

    def test_atom_types(self):
        # Check proper return of atom types
        for type_count in range(0, 5):
            self.assertEqual(type_count, crystal.Cell(numpy.eye(3), [numpy.eye(3)] * type_count).atom_types)

    def test_atom_counts(self):
        # Check proper return of atom counts
        for atom_count in range(1, 5):
            self.assertEqual([atom_count, atom_count + 1], crystal.Cell(numpy.eye(3), \
                [numpy.zeros((atom_count, 3)), numpy.zeros((atom_count + 1, 3))]).atom_counts)

    def test_atom_count(self):
        # Check proper return of individual atom counts and name resolution
        for atom_types in range(1, 5):
            for atom_count in range(1, 5):
                cell = crystal.Cell(numpy.eye(3), [numpy.zeros((atom_count + index, 3)) \
                    for index in range(atom_types)], [str(index) for index in range(atom_types)])
                for index in range(atom_count):
                    for atom_type in range(atom_types):
                        self.assertEqual(cell.atom_count(atom_type), atom_count + atom_type)
                        self.assertEqual(cell.atom_count(atom_type), cell.atom_count(str(atom_type)))

    def test_atom_lists(self):
        # Check proper return of all atom lists
        atom_lists = [numpy.linspace(index, index ** 2, 9).reshape(3, 3) for index in range(5)]
        cell = crystal.Cell(numpy.eye(3), atom_lists)
        for index in range(cell.atom_types):
            self.assertTrue(numpy.all(cell.atom_lists[index] == atom_lists[index]))

    def test_atoms(self):
        # Check proper return of individual atom lists
        atom_lists = [numpy.linspace(index, index ** 2, 9).reshape(3, 3) for index in range(5)]
        cell = crystal.Cell(numpy.eye(3), atom_lists, [str(index) for index in range(len(atom_lists))])
        for index in range(cell.atom_types):
            self.assertTrue(numpy.all(cell.atoms(index) == atom_lists[index]))
            self.assertTrue(numpy.all(cell.atoms(index) == cell.atoms(str(index))))

    def test_atom(self):
        # Check proper return of individual atoms
        atom_lists = [numpy.linspace(index, index ** 2, 9).reshape(3, 3) for index in range(5)]
        cell = crystal.Cell(numpy.eye(3), atom_lists, [str(index) for index in range(len(atom_lists))])
        for type_index in range(cell.atom_types):
            for atom_index in range(cell.atom_count(type_index)):
                self.assertTrue(numpy.all(cell.atom(type_index, atom_index) == atom_lists[type_index][atom_index]))
                self.assertTrue(numpy.all(cell.atom(type_index, atom_index) == cell.atom(str(type_index), atom_index)))

    def test_names(self):
        # Test proper return of names
        names = ["Q", "R", "S", "T", "U", "V"]
        cell = crystal.Cell(numpy.eye(3), [numpy.eye(3) for index in range(len(names))], names)
        self.assertEqual(cell.names, names)

    def test_name(self):
        # Test proper resolution of indices into names
        names = ["Q", "R", "S", "T", "U", "V"]
        cell = crystal.Cell(numpy.eye(3), [numpy.eye(3) for index in range(len(names))], names)
        for index in range(len(names)):
            self.assertEqual(cell.name(index), names[index])

    def test_index(self):
        # Test proper resolution of names into indices
        names = ["Q", "R", "S", "T", "U", "V"]
        cell = crystal.Cell(numpy.eye(3), [numpy.eye(3) for index in range(len(names))], names)
        for index, name in enumerate(names):
            self.assertEqual(index, cell.index(name))

    def test_measure_to_bad_spec(self):
        # measure_to should never affect evaluation results and needs one or two cutoffs
        cell = crystal.Cell(numpy.eye(3), [0.5 * numpy.ones((1, 3))])
        rdf_before = cell.rdf(0, 0, 3)
        with self.assertRaises(Exception):
            cell.measure_to()
        rdf_after = cell.rdf(0, 0, 3)
        self.assertEqual(rdf_before, rdf_after)

    def test_measure_to_shell(self):
        # measure_to should never affect evaluation results
        for shell_cutoff in range(0, 6):
            cell = crystal.Cell(numpy.eye(3), [0.5 * numpy.ones((1, 3))])
            rdf_before = cell.rdf(0, 0, 3)
            cell.measure_to(shell_cutoff=shell_cutoff)
            rdf_after = cell.rdf(0, 0, 3)
            self.assertEqual(rdf_before, rdf_after)

    def test_measure_to_distance(self):
        # measure_to should never affect evaluation results
        for distance_cutoff in range(0, 6):
            cell = crystal.Cell(numpy.eye(3), [0.5 * numpy.ones((1, 3))])
            rdf_before = cell.rdf(0, 0, 3)
            cell.measure_to(distance_cutoff=distance_cutoff)
            rdf_after = cell.rdf(0, 0, 3)
            self.assertEqual(rdf_before, rdf_after)

    def test_rdf_persistence(self):
        # Writing and reading RDFs should never affect data
        for distance_cutoff in range(0, 6):
            cell = crystal.Cell(numpy.eye(3), [0.5 * numpy.ones((1, 3))])
            rdf_before = cell.rdf(0, 0, 3)
            cell.measure_to(distance_cutoff=distance_cutoff)
            rdf_data = io.BytesIO()
            cell.write_rdf(rdf_data)
            rdf_data.seek(0)
            cell.read_rdf(rdf_data)
            rdf_after = cell.rdf(0, 0, 3)
            self.assertEqual(rdf_before, rdf_after)

    def test_rdf_known_2d(self):
        # Check a known configuration for consistency
        cell2 = crystal.Cell(numpy.eye(2), [numpy.zeros((1, 2))])
        rdf2 = cell2.rdf(0, 0, 1.5)
        self.assertEqual(len(rdf2), 2)
        rdf2 = sorted(rdf2.items(), key=lambda pair: pair[0])
        self.assertAlmostEqual(rdf2[0][0], 1)
        self.assertAlmostEqual(rdf2[1][0], 2.0 ** 0.5)
        self.assertEqual(rdf2[0][1], 4)
        self.assertEqual(rdf2[1][1], 4)

    def test_rdf_known_3d(self):
        # Check a known configuration for consistency
        cell3 = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3))])
        rdf3 = cell3.rdf(0, 0, 1.8)
        self.assertEqual(len(rdf3), 3)
        rdf3 = sorted(rdf3.items(), key=lambda pair: pair[0])
        self.assertAlmostEqual(rdf3[0][0], 1)
        self.assertAlmostEqual(rdf3[1][0], 2.0 ** 0.5)
        self.assertAlmostEqual(rdf3[2][0], 3.0 ** 0.5)
        self.assertEqual(rdf3[0][1], 6)
        self.assertEqual(rdf3[1][1], 12)
        self.assertEqual(rdf3[2][1], 8)

    def test_rdf_swap(self):
        # Check type swapping returns identical RDFs
        cell = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), numpy.array([[0.3] * 3, [0.65] * 3])])
        self.assertNotEqual(cell.rdf(0, 0, 6), cell.rdf(1, 1, 6))
        self.assertNotEqual(cell.rdf(0, 0, 6), cell.rdf(0, 1, 6))
        self.assertEqual(cell.rdf(0, 1, 6), cell.rdf(1, 0, 6))

    def test_contact(self):
        # Check that minimum contact distance is always returned
        cell = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), numpy.array([[0.3] * 3, [0.65] * 3])])
        self.assertEqual(min(cell.rdf(0, 0, 6)), cell.contact(0, 0))
        self.assertEqual(min(cell.rdf(0, 1, 6)), cell.contact(0, 1))
        self.assertEqual(min(cell.rdf(1, 0, 6)), cell.contact(1, 0))
        self.assertEqual(min(cell.rdf(1, 1, 6)), cell.contact(1, 1))

    def test_scale_factor(self):
        # Check scan behavior over range of radii
        cell = crystal.Cell(numpy.eye(2), [numpy.zeros((1, 2)), 0.5 * numpy.ones((1, 2))])
        critical_ratio = (2.0 ** 0.5) - 1
        # A very large compared to B, scaling B has no effect
        factors = [cell.scale_factor([1, value]) for value in critical_ratio * numpy.linspace(0.1, 1, 10)]
        self.assertTrue(numpy.all(numpy.isclose(numpy.diff(factors), 0)))
        # B very large compared to A, scaling A has no effect
        factors = [cell.scale_factor([value, 1]) for value in critical_ratio * numpy.linspace(0.1, 1, 10)]
        self.assertTrue(numpy.all(numpy.isclose(numpy.diff(factors), 0)))
        # A and B contact each other instead of themselves, scaling has an effect
        factors = [cell.scale_factor([1, value]) for value in numpy.linspace(critical_ratio, 1 / critical_ratio, 10)]
        self.assertFalse(numpy.any(numpy.isclose(numpy.diff(factors), 0)))

class CellToolsTests(unittest.TestCase):

    def test_similar(self):
        reference = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), 0.5 * numpy.ones((2, 3))])
        identical = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), 0.5 * numpy.ones((2, 3))])
        near_identical = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)) + 1e-8, 0.5 * numpy.ones((2, 3))])
        different_numbers = crystal.Cell(numpy.eye(3), [numpy.ones((1, 3)), numpy.zeros((2, 3))])
        different_shapes = crystal.Cell(numpy.eye(2), [numpy.zeros((1, 2)), 0.5 * numpy.ones((3, 2))])

        self.assertTrue(crystal.CellTools.similar(reference, reference))
        self.assertTrue(crystal.CellTools.similar(reference, identical))
        self.assertTrue(crystal.CellTools.similar(reference, near_identical))
        self.assertFalse(crystal.CellTools.similar(reference, near_identical, tolerance=1e-10))
        self.assertFalse(crystal.CellTools.similar(reference, different_numbers))
        self.assertFalse(crystal.CellTools.similar(reference, different_shapes))

    def test_identical(self):
        reference = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), 0.5 * numpy.ones((2, 3))])
        identical = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), 0.5 * numpy.ones((2, 3))])
        near_identical = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)) + 1e-8, 0.5 * numpy.ones((2, 3))])
        different_numbers = crystal.Cell(numpy.eye(3), [numpy.ones((1, 3)), numpy.zeros((2, 3))])
        different_shapes = crystal.Cell(numpy.eye(2), [numpy.zeros((1, 2)), 0.5 * numpy.ones((3, 2))])

        self.assertTrue(crystal.CellTools.identical(reference, reference))
        self.assertTrue(crystal.CellTools.identical(reference, identical))
        self.assertFalse(crystal.CellTools.identical(reference, near_identical))
        self.assertFalse(crystal.CellTools.identical(reference, different_numbers))
        self.assertFalse(crystal.CellTools.identical(reference, different_shapes))

    def test_rename(self):
        cell = crystal.Cell(numpy.eye(2), [numpy.ones((index + 1, 2)) for index in range(3)], ["A", "B", "C"])

        # Check for good renames
        self.assertEqual(crystal.CellTools.rename(cell, {"A": "X", "B": "Y", "C": "Z"}).names, ["X", "Y", "Z"])
        self.assertEqual(crystal.CellTools.rename(cell, {"A": 2, "B": 1, "C": 0}).names, ["C", "B", "A"])
        self.assertEqual(crystal.CellTools.rename(cell, {1: "X", 0: 1, "C": 2}).names, ["B", "X", "C"])

        # Check for no atom list reordering
        self.assertTrue(all(numpy.all(atom_list == cell.atoms(index)) \
            for index, atom_list in enumerate(crystal.CellTools.rename(cell, {"A": "X", "B": "Y", "C": "Z"}).atom_lists)))
        self.assertTrue(all(numpy.all(atom_list == cell.atoms(index)) \
            for index, atom_list in enumerate(crystal.CellTools.rename(cell, {"A": 2, "B": 1, "C": 0}).atom_lists)))
        self.assertTrue(all(numpy.all(atom_list == cell.atoms(index)) \
            for index, atom_list in enumerate(crystal.CellTools.rename(cell, {1: "X", 0: 1, "C": 2}).atom_lists)))

        # Check for errors on invalid renames
        with self.assertRaises(Exception):
            crystal.CellTools.rename(cell, {"A": "X", "B": "Y", "C": "Y"})
        with self.assertRaises(Exception):
            crystal.CellTools.rename(cell, {"A": 1, "B": 2, "C": 3})

    def test_reassign(self):
        cell = crystal.Cell(numpy.eye(2), [numpy.ones((index + 1, 2)) for index in range(3)], ["A", "B", "C"])

        # Check for good reassignments
        self.assertEqual(crystal.CellTools.reassign(cell, {"A": "B", "B": "C", "C": "A"}).names, ["C", "A", "B"])
        self.assertEqual(crystal.CellTools.reassign(cell, {0: 1, 1: 1, 2: 0}).names, ["C", "A"])
        self.assertEqual(crystal.CellTools.reassign(cell, {1: "A", "C": 1}).names, ["B", "C"])

        # Check for proper atom list reordering and merging
        self.assertTrue(all(numpy.all(atom_list == cell.atoms([2, 1, 0][index])) \
            for index, atom_list in enumerate(crystal.CellTools.reassign(cell, {"A": "B", "B": "C", "C": "A"}).atom_lists)))
        self.assertTrue(all(numpy.all(atom_list == [cell.atoms(2), numpy.concatenate([cell.atoms(0), cell.atoms(1)])][index]) \
            for index, atom_list in enumerate(crystal.CellTools.reassign(cell, {0: 1, 1: 1, 2: 0}).atom_lists)))
        self.assertTrue(all(numpy.all(atom_list == cell.atoms([1, 2][index])) \
            for index, atom_list in enumerate(crystal.CellTools.reassign(cell, {1: "A", "C": 1}).atom_lists)))

        # Check for errors on invalid reassignments
        with self.assertRaises(Exception):
            crystal.CellTools.reassign(cell, {"A": "B", "B": "B", "C": 3})
        with self.assertRaises(Exception):
            crystal.CellTools.reassign(cell, {"X": 0})
        with self.assertRaises(Exception):
            crystal.CellTools.reassign(cell, {0: "A", 1: "Y", 2: "A"})

    def test_scale(self):
        cell = crystal.Cell(numpy.eye(2), [numpy.zeros((1, 2)), 0.5 * numpy.ones((1, 2))])

        # Total direct scale
        self.assertTrue(numpy.all(numpy.isclose(cell.vectors * 2, crystal.CellTools.scale(cell, cell.vectors * 2).vectors)))
        self.assertTrue(all(numpy.all(numpy.isclose(cell.atoms(index) * 2, crystal.CellTools.scale(cell, cell.vectors * 2).atoms(index))) for index in range(2)))

        # Don't move vectors
        self.assertTrue(numpy.all(numpy.isclose(cell.vectors, crystal.CellTools.scale(cell, cell.vectors * 2, move_vectors=False).vectors)))
        self.assertTrue(all(numpy.all(numpy.isclose(cell.atoms(index) * 2, crystal.CellTools.scale(cell, cell.vectors * 2, move_vectors=False).atoms(index))) for index in range(2)))

        # Don't move atoms
        self.assertTrue(numpy.all(numpy.isclose(cell.vectors * 2, crystal.CellTools.scale(cell, cell.vectors * 2, move_atoms=False).vectors)))
        self.assertTrue(all(numpy.all(numpy.isclose(cell.atoms(index), crystal.CellTools.scale(cell, cell.vectors * 2, move_atoms=False).atoms(index))) for index in range(2)))

    def test_scale_shear(self):
        # Shearing
        cell = crystal.Cell(numpy.array([[1, 0.01, 0.02], [0.03, 1, 0.04], [0.05, 0.06, 1]]), \
            [numpy.array([[0.25, 0.26, 0.27], [0.55, 0.56, 0.57]])])
        new_vectors = numpy.array([[1, 0.05, 0.10], [0.15, 2, 0.20], [0.25, 0.30, 4]])
        new_cell = crystal.CellTools.scale(cell, new_vectors)
        self.assertTrue(numpy.all(numpy.isclose(new_cell.vectors, new_vectors)))
        self.assertTrue(numpy.all(numpy.isclose(numpy.dot(new_vectors.T, numpy.dot(numpy.linalg.inv(cell.vectors.T), cell.atoms(0).T)), new_cell.atoms(0).T)))

    def test_normalize(self):
        from paccs import potential
        from potential import EvaluateTests

        for case in EvaluateTests._cases:
            cell = crystal.CellTools.wrap(case[0])

            # Make sure row vector matrix is lower triangular
            new_cell = crystal.CellTools.normalize(cell)
            self.assertTrue(numpy.all(numpy.isclose(numpy.tril(new_cell.vectors), new_cell.vectors)))

            # Make sure that geometric parameters have not changed
            self.assertAlmostEqual(cell.enclosed, new_cell.enclosed)
            self.assertAlmostEqual(cell.surface, new_cell.surface)
            self.assertAlmostEqual(cell.distortion_factor, new_cell.distortion_factor)

            # Make sure that energy didn't change in the normalization
            self.assertAlmostEqual(potential._evaluate_fast(cell, case[1], 2.54321)[0], \
                potential._evaluate_fast(new_cell, case[1], 2.54321)[0])

    def test_wrap_right(self):
        # Check cubical box
        cell = crystal.Cell(numpy.eye(3), [numpy.array([[0.1, 0.1, 0.1], [1.2, 0.2, 0.2], \
            [0.3, 1.3, 0.3], [0.4, 0.4, 1.4], [-0.5, 0.5, 0.5], [0.6, -0.4, 0.6], [0.7, -0.3, 0.7]])])
        cell_normal = crystal.Cell(numpy.eye(3), [numpy.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], \
            [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.7, 0.7, 0.7]])])
        self.assertTrue(numpy.all(numpy.isclose(cell_normal.vectors, crystal.CellTools.wrap(cell_normal).vectors)))
        self.assertTrue(numpy.all(numpy.isclose(cell_normal.atoms(0), crystal.CellTools.wrap(cell_normal).atoms(0))))
        self.assertTrue(numpy.all(numpy.isclose(cell_normal.vectors, crystal.CellTools.wrap(cell).vectors)))
        self.assertTrue(numpy.all(numpy.isclose(cell_normal.atoms(0), crystal.CellTools.wrap(cell).atoms(0))))

    def test_wrap_sheared(self):
        # Check sheared box
        cell = crystal.Cell(numpy.eye(3), [numpy.array([[0.1, 0.1, 0.1], [1.2, 0.2, 0.2], \
            [0.3, 1.3, 0.3], [0.4, 0.4, 1.4], [-0.5, 0.5, 0.5], [0.6, -0.4, 0.6], [0.7, -0.3, 0.7]])])
        cell_normal = crystal.Cell(numpy.eye(3), [numpy.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], \
            [0.3, 0.3, 0.3], [0.4, 0.4, 0.4], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6], [0.7, 0.7, 0.7]])])
        vectors = numpy.array([[1.1, 0.2, 0.3], [0.4, 0.5, 0.6], [-0.1, 1.1, 2.1]])
        cell = crystal.CellTools.scale(cell, vectors)
        cell_normal = crystal.CellTools.scale(cell_normal, vectors)
        self.assertTrue(numpy.all(numpy.isclose(cell_normal.vectors, crystal.CellTools.wrap(cell_normal).vectors)))
        self.assertTrue(numpy.all(numpy.isclose(cell_normal.atoms(0), crystal.CellTools.wrap(cell_normal).atoms(0))))
        self.assertTrue(numpy.all(numpy.isclose(cell_normal.vectors, crystal.CellTools.wrap(cell).vectors)))
        self.assertTrue(numpy.all(numpy.isclose(cell_normal.atoms(0), crystal.CellTools.wrap(cell).atoms(0))))

    def test_condense_direct(self):
        # Try with a cube
        cell_before = crystal.Cell(numpy.eye(3), [numpy.array([[0.5, 0.5, 0.5], [0.5 + 1e-7, 0.5 + 1e-7, 0.5 - 1e-7], \
            [0.5 + 1e-6, 0.5 + 1e-6, 0.5 + 1e-6]]), numpy.array([[0.5, 0.5, 0.5], [0.5 + 1e-7, 0.5 + 1e-7, 0.5 + 1e-7]])])
        cell_reference = crystal.Cell(numpy.eye(3), [numpy.array([[0.5, 0.5, 0.5], \
            [0.5 + 1e-6, 0.5 + 1e-6, 0.5 + 1e-6]]), numpy.array([[0.5, 0.5, 0.5]])])
        cell_condensed = crystal.CellTools.condense(cell_before, tolerance=5e-7)
        self.assertTrue(numpy.all(numpy.isclose(cell_condensed.vectors, cell_reference.vectors)))
        self.assertTrue(numpy.all(numpy.isclose(cell_condensed.atoms(0), cell_reference.atoms(0))))
        self.assertTrue(numpy.all(numpy.isclose(cell_condensed.atoms(1), cell_reference.atoms(1))))

        # Shear and make sure things still work
        vectors = numpy.array([[1.0, 0.0, 0.1], [0.2, 0.4, 2.0], [0.5, 1.1, -0.2]])
        cell_before = crystal.CellTools.scale(cell_before, vectors)
        cell_reference = crystal.CellTools.scale(cell_reference, vectors)
        cell_condensed = crystal.CellTools.condense(cell_before, tolerance=5e-7)
        self.assertTrue(numpy.all(numpy.isclose(cell_condensed.vectors, cell_reference.vectors)))
        self.assertTrue(numpy.all(numpy.isclose(cell_condensed.atoms(0), cell_reference.atoms(0))))
        self.assertTrue(numpy.all(numpy.isclose(cell_condensed.atoms(1), cell_reference.atoms(1))))

    def test_condense_wrap(self):
        # Try with a cube
        unit_cell = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), 0.5 * numpy.ones((1, 3))])
        supercell = crystal.CellTools.scale(crystal.CellTools.tile(unit_cell, (1, 2, 3)), unit_cell.vectors, move_atoms=False)
        condensed = crystal.CellTools.condense(supercell)
        self.assertTrue(numpy.all(numpy.isclose(condensed.vectors, unit_cell.vectors)))
        self.assertTrue(numpy.all(numpy.isclose(condensed.atoms(0), unit_cell.atoms(0))))
        self.assertTrue(numpy.all(numpy.isclose(condensed.atoms(1), unit_cell.atoms(1))))

        # Shear and make sure things still work
        unit_cell = crystal.CellTools.scale(unit_cell, numpy.array([[1.0, 0.0, 0.1], [0.2, 0.4, 2.0], [0.5, 1.1, -0.2]]))
        supercell = crystal.CellTools.scale(crystal.CellTools.tile(unit_cell, (1, 2, 3)), unit_cell.vectors, move_atoms=False)
        condensed = crystal.CellTools.condense(supercell)
        self.assertTrue(numpy.all(numpy.isclose(condensed.vectors, unit_cell.vectors)))
        self.assertTrue(numpy.all(numpy.isclose(condensed.atoms(0), unit_cell.atoms(0))))
        self.assertTrue(numpy.all(numpy.isclose(condensed.atoms(1), unit_cell.atoms(1))))

    def test_shift(self):
        cell = crystal.Cell(numpy.eye(3), [numpy.concatenate([0.05 * numpy.ones((1, 3)), \
            0.3 * numpy.ones((1, 3))]), (0.55 * numpy.ones((1, 3))) + numpy.array([[0.03, 0.04, 0.07]])])
        cell_default = crystal.Cell(numpy.eye(3), [numpy.concatenate([numpy.zeros((1, 3)), \
            0.25 * numpy.ones((1, 3))]), (0.5 * numpy.ones((1, 3))) + numpy.array([[0.03, 0.04, 0.07]])])
        cell_selected = crystal.Cell(numpy.eye(3), [numpy.concatenate([(-0.25 * numpy.ones((1, 3))) - \
            numpy.array([[0.03, 0.04, 0.07]]), -numpy.array([[0.03, 0.04, 0.07]])]), 0.25 * numpy.ones((1, 3))])
        cell_shift_default = crystal.CellTools.shift(cell)
        cell_shift_selected = crystal.CellTools.shift(cell, 0, 1, -numpy.array([[0.03, 0.04, 0.07]]))
        self.assertTrue(numpy.all(numpy.isclose(cell_default.vectors, cell_shift_default.vectors)))
        self.assertTrue(numpy.all(numpy.isclose(cell_default.atoms(0), cell_shift_default.atoms(0))))
        self.assertTrue(numpy.all(numpy.isclose(cell_default.atoms(1), cell_shift_default.atoms(1))))
        self.assertTrue(numpy.all(numpy.isclose(cell_selected.vectors, cell_shift_selected.vectors)))
        self.assertTrue(numpy.all(numpy.isclose(cell_selected.atoms(0), cell_shift_selected.atoms(0))))
        self.assertTrue(numpy.all(numpy.isclose(cell_selected.atoms(1), cell_shift_selected.atoms(1))))

    def test_tile(self):
        CsCl = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), 0.5 * numpy.ones((1, 3))])
        CsCl_unit = crystal.CellTools.tile(CsCl, (1, 1, 1), (1, 1))
        CsCl_supercell = crystal.CellTools.tile(CsCl, (1, 1, 1))
        CsCl_unit_condense = crystal.CellTools.condense(CsCl_unit)
        CsCl_supercell_condense = crystal.CellTools.condense(crystal.CellTools.scale(CsCl_supercell, CsCl.vectors, move_atoms=False))

        self.assertTrue(numpy.all(numpy.isclose(CsCl.vectors, CsCl_unit_condense.vectors)))
        self.assertTrue(numpy.all(numpy.isclose(CsCl.atoms(0), CsCl_unit_condense.atoms(0))))
        self.assertTrue(numpy.all(numpy.isclose(CsCl.atoms(1), CsCl_unit_condense.atoms(1))))
        self.assertTrue(numpy.all(numpy.isclose(CsCl.vectors, CsCl_supercell_condense.vectors)))
        self.assertTrue(numpy.all(numpy.isclose(CsCl.atoms(0), CsCl_supercell_condense.atoms(0))))
        self.assertTrue(numpy.all(numpy.isclose(CsCl.atoms(1), CsCl_supercell_condense.atoms(1))))

    def test_reduce(self):
        from potential import EvaluateTests
        from paccs import potential

        for case in EvaluateTests._cases:
            cell = case[0]
            new_cell = crystal.CellTools.reduce(cell)

            # Make sure row vector matrix is lower triangular
            self.assertTrue(numpy.all(numpy.isclose(numpy.tril(new_cell.vectors), new_cell.vectors)))

            # Make sure that geometric parameters have changed within proper bounds
            self.assertAlmostEqual(cell.enclosed, new_cell.enclosed)
            self.assertTrue(new_cell.surface < cell.surface \
                or numpy.isclose(new_cell.surface, cell.surface))
            self.assertTrue(new_cell.distortion_factor < cell.distortion_factor \
                or numpy.isclose(new_cell.distortion_factor, cell.distortion_factor))

            # Make sure automatic shifting and wrapping have worked
            new_cell_shift = crystal.CellTools.shift(new_cell)
            new_cell_wrap = crystal.CellTools.wrap(new_cell)
            self.assertTrue(numpy.all(numpy.isclose(new_cell.vectors, new_cell_shift.vectors)))
            self.assertTrue(numpy.all(numpy.isclose(new_cell.vectors, new_cell_wrap.vectors)))
            for index in range(new_cell.atom_types):
                self.assertTrue(numpy.all(numpy.isclose(new_cell.atoms(index), new_cell_shift.atoms(index))))
                self.assertTrue(numpy.all(numpy.isclose(new_cell.atoms(index), new_cell_wrap.atoms(index))))

            # Make sure that energy didn't change in the reduction
            self.assertAlmostEqual(potential._evaluate_fast(cell, case[1], 2.54321)[0], \
                potential._evaluate_fast(new_cell, case[1], 2.54321)[0])

    # def test_energy(self): See the tests for paccs.potential._evaluate_fast

class CellCodecsTests(unittest.TestCase):

    def test_cell_consistency(self):
        # Check round-trip consistency
        from potential import EvaluateTests
        for case in EvaluateTests._cases:
            cell = case[0]
            cell_data = io.StringIO()
            crystal.CellCodecs.write_cell(cell, cell_data)
            cell_data.seek(0)
            new_cell = crystal.CellCodecs.read_cell(cell_data)
            self.assertTrue(numpy.all(cell.vectors == new_cell.vectors))
            for type_index in range(cell.atom_types):
                self.assertTrue(numpy.all(cell.atoms(type_index) == new_cell.atoms(type_index)))
            self.assertEqual(cell.names, new_cell.names)

    def test_xyz_stable(self):
        # TODO: tests only no crashing; should also test output
        from potential import EvaluateTests
        for case in EvaluateTests._cases:
            cell = case[0]
            cell_data = io.StringIO()
            crystal.CellCodecs.write_xyz(cell, cell_data)

    def test_lammps_stable(self):
        # TODO: tests only no crashing; should also test output
        from potential import EvaluateTests
        for case in EvaluateTests._cases:
            cell = case[0]
            cell_data = io.StringIO()
            crystal.CellCodecs.write_lammps(cell, cell_data)
