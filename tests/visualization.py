#!/usr/bin/env python

import itertools
import numpy
import unittest

from paccs import crystal
from paccs import visualization

_cells = [
        crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3))]),
        crystal.Cell(numpy.eye(2), [numpy.zeros((1, 2)), 0.5 * numpy.ones((1, 2))]),
        crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), 0.5 * numpy.ones((1, 3))]),
        crystal.Cell(numpy.eye(3), [numpy.concatenate([numpy.zeros((1, 3)), numpy.ones((1, 3))])])]

class Tests(unittest.TestCase):

    def test_cell(self):
        for cell in _cells:
            coordinates, colors, sizes, types, boxes = \
                visualization._cell(cell, [1] * cell.dimensions, [i + 1 for i in range(cell.atom_types)])

            # Make sure output is reasonable
            count = sum(cell.atom_counts)
            self.assertEqual(coordinates.shape[0], count)
            self.assertEqual(colors.shape[0], count)
            self.assertEqual(sizes.shape[0], count)
            self.assertEqual(types.shape[0], count)

    def test_cell_types(self):
        for cell in _cells:
            coordinates, colors, sizes, types, boxes = \
                visualization._cell(cell, [1] * cell.dimensions, [i + 1 for i in range(cell.atom_types)])

            # Make sure types are assigned properly
            for type_index in range(cell.atom_types):
                colors_type = set(colors[types == type_index])
                sizes_type = set(sizes[types == type_index])

                self.assertEqual(len(colors_type), 1)
                self.assertEqual(len(sizes_type), 1)

            for type_index_1 in range(cell.atom_types):
                for type_index_2 in range(cell.atom_types):
                    colors_1 = set(colors[types == type_index_1]).pop()
                    colors_2 = set(colors[types == type_index_2]).pop()
                    sizes_1 = set(sizes[types == type_index_1]).pop()
                    sizes_2 = set(sizes[types == type_index_2]).pop()

                    self.assertEqual(colors_1 == colors_2, type_index_1 == type_index_2)
                    self.assertEqual(sizes_1 == sizes_2, type_index_1 == type_index_2)

    def test_cell_repeats(self):
        for cell in _cells:
            for repeats in itertools.product(*(range(1, 4) for index in range(cell.dimensions))):
                coordinates, colors, sizes, types, boxes = \
                    visualization._cell(cell, repeats, [i + 1 for i in range(cell.atom_types)])

                # Make sure repeats are being performed properly
                count = sum(cell.atom_counts) * numpy.product(repeats)
                self.assertEqual(coordinates.shape[0], count)
                self.assertEqual(colors.shape[0], count)
                self.assertEqual(sizes.shape[0], count)
                self.assertEqual(types.shape[0], count)

if __name__ == "__main__":
    print("Executing live visualization tests.")

    def read_cell(path):
        with open(path, "r") as file:
            return crystal.CellCodecs.read_cell(file)

    for renderer in [visualization.cell_mayavi, visualization.cell_plotly]:
        print("Using renderer {}".format(renderer))

        print("2D square lattice with varied radii")
        cell = read_cell("data/vis_test_1.cell")
        renderer(cell, (1, 1), (1, 2))

        print("3D CsCl lattice with 4x3x2 repeats")
        cell = read_cell("data/vis_test_2.cell")
        renderer(cell, (4, 3, 2), (1, 1))

        print("3D ternary system with full unit cell display")
        cell = read_cell("data/vis_test_3.cell")
        renderer(cell, (1, 1, 1), (1.69, 1.3, 1), partial=True)

        print("2D hexagonal lattice with high resolution")
        cell = read_cell("data/vis_test_4.cell")
        renderer(cell, (1, 1), (1, 1), resolution=(48, 48))

    print("Displaying wallpaper group tilings")
    renderer = visualization.cell_mayavi
    for group in range(17):
        print("Group {}".format(group + 1))
        cell = read_cell("data/group_{}.cell".format(group + 1))
        renderer(cell, (3, 3), (1, 1, 1), supercell_box=False, cell_boxes=False)
