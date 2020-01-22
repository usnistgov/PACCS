"""
Contains routines to perform manipulations on crystal cells.
"""

import cython
import collections
import itertools
import numpy
import pickle
import scipy.spatial.distance
import scipy

cimport cython
cimport numpy
cimport libc.math

@cython.final(True)
cdef class Cell:
    """
    Creates a cell with the specified parameters.

    Parameters
    ----------
    vectors : numpy.ndarray
        Cell vectors as a square matrix of row vectors.  The size of this matrix
        determines the number of dimensions of the cell, which must be at least 2.
    atom_lists : list(numpy.ndarray)
        Atom coordinates as matrices of row vectors.  Each matrix in the list should
        specify the coordinates of one type of atoms.
    names : list(str)
        Names of types of atoms (letters will be used if no names are provided).

    Raises
    ------
    ValueError
        Invalid or inconsistent dimensions or values.
    """

    cdef numpy.ndarray __vectors
    cdef long __dimensions
    cdef list __atom_lists
    cdef long __atom_types
    cdef list __names
    cdef list __rdfs
    cdef long __shells_scanned

    def __init__(self, vectors, atom_lists, names=None):
        # Process the vector list and retrieve the number of dimensions
        self.__vectors = numpy.array(vectors, dtype=float, order="C")
        if self.__vectors.ndim != 2 or self.__vectors.shape[0] != self.__vectors.shape[1]:
            raise ValueError("vectors must be provided as a square matrix")
        self.__dimensions = self.__vectors.shape[1]
        if self.__dimensions < 2:
            raise ValueError("at least two dimensions must be present")

        # Process the atom lists containing coordinates of atoms of different types
        self.__atom_lists = []
        for atom_list in atom_lists:
            atom_list = numpy.array(atom_list, dtype=float, order="C")
            if atom_list.ndim != 2 or atom_list.shape[1] != self.__dimensions:
                raise ValueError("atom coordinates must be consistent with cell dimensions")
            self.__atom_lists.append(atom_list)
        self.__atom_types = len(self.__atom_lists)

        # Process the list of names
        if names is None:
            names = [chr(ord("A") + index) for index in range(self.__atom_types)]
        self.__names = list(names)
        if len(set(self.__names)) != len(self.__names):
            raise ValueError("all atom type names must be unique")
        if len(self.__names) != self.__atom_types:
            raise ValueError("number of atom type names must match number of atom types")

        # Set up RDF array
        self.__rdfs = [[collections.Counter() \
            for target_type_index in range(source_type_index + 1)] \
            for source_type_index in range(self.__atom_types)]
        # Store number of shells scanned (0 = current image alone)
        self.__shells_scanned = -1

    def __repr__(self):
        return "<paccs.crystal.Cell at 0x{:x} in {}D{}>".format(id(self), self.__dimensions, \
            " with {}".format(", ".join("{} {}".format(self.__atom_lists[type_index].shape[0], self.__names[type_index]) \
            for type_index in range(len(self.__atom_lists)))) if self.__atom_types else "")

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return NotImplemented
        return CellTools.identical(self, other)

    def __ne__(self, other):
        if not isinstance(other, Cell):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self):
        return hash((bytes(self.__vectors.data), \
            tuple(bytes(atom_list.data) for atom_list in self.__atom_lists), tuple(self.__names)))

    @property
    def dimensions(self):
        """
        Retrieves the number of spatial dimensions of the cell.

        Returns
        -------
        int
            The number of dimensions.
        """

        return self.__dimensions

    @property
    def vectors(self):
        """
        Retrieves the cell vectors.

        Returns
        -------
        numpy.ndarray
            Cell vectors as a matrix of row vectors.
        """

        return self.__vectors.copy()

    def vector(self, vector_index):
        """
        Retrieves a cell vector.

        Parameters
        ----------
        vector_index : int
            Index of the cell vector to retrieve.

        Returns
        -------
        numpy.ndarray
            Row vector requested.
        """

        return self.__vectors[vector_index].copy()

    @staticmethod
    def __space(vectors):
        """
        Retrieves the space enclosed by a parallelotope.

        Parameters
        ----------
        vectors : numpy.ndarray
            A matrix of row vectors defining the edges of the parallelotope, which
            need not be square.

        Returns
        -------
        float
            The space (area for a two-dimensional parallelotope, volume for a three-
            dimensional parallelotope) enclosed.
        """

        return numpy.sqrt(numpy.linalg.det(numpy.dot(vectors, vectors.T)))

    @property
    def enclosed(self):
        """
        Retrieves the space enclosed by the cell.  If the number of dimensions for
        the cell is N, this quantity is N-dimensional.

        Returns
        -------
        float
            The space (area for a two-dimensional cell, volume for a three-
            dimensional cell) enclosed.
        """

        return Cell.__space(self.__vectors)

    @property
    def surface(self):
        """
        Retrieves the surface space of the cell.  If the number of dimensions for
        the cell is N, this quantity is (N-1)-dimensional.

        Returns
        -------
        float
            The space (length for a two-dimensional cell, area for a three-
            dimensional cell) of the surface.
        """

        return 2 * sum(Cell.__space(numpy.array([self.vectors[vector_index] \
            for vector_index in range(self.__dimensions) \
            if vector_index != dimension_index])) \
            for dimension_index in range(self.__dimensions))

    @property
    def distortion_factor(self):
        """
        Retrieves the normalized distortion factor of the cell.  This parameter
        is discussed in *de Graaf et al.* (`doi:10.1063/1.4767529
        <https://dx.doi.org/10.1063/1.4767529>`_).

        Returns
        -------
        float
            The distortion factor (1 for a cell whose vectors are all orthogonal
            to each other and of equal length, or greater than 1 otherwise).
        """

        # Calculate the distortion factor and divide by the normalization factor
        # of 2N such that the lower bound on the result is always 1
        average_length = sum(numpy.linalg.norm(self.__vectors[dimension_index]) \
            for dimension_index in range(self.__dimensions)) / self.__dimensions
        distortion_factor = average_length * self.surface / (2 * self.enclosed * self.__dimensions)
        return max(1.0, distortion_factor) # in case of transient rounding errors

    @property
    def normals(self):
        """
        Retrieves normals to faces of cell planes.  For a cell with a diagonal
        vector matrix, these normals will point in the same direction as the vectors.

        Returns
        -------
        numpy.ndarray
            A matrix of normals as row vectors, beginning with the normal of the
            plane such that all but the first of the cell vectors are coplanar.
        """

        # Calculate normals to cell planes
        return numpy.array([numpy.array([numpy.linalg.det( \
            numpy.array([self.__vectors[index] for index in range(self.__dimensions) if index != dimension_index]) \
            [:, [index for index in range(self.__dimensions) if index != component_index]]) * (-1 if component_index % 2 else 1) \
            for component_index in range(self.__dimensions)]) * (-1 if dimension_index % 2 else 1) \
            for dimension_index in range(self.__dimensions)])

    @property
    def atom_types(self):
        """
        Retrieves the number of atom types.

        Returns
        -------
        int
            Number of types.
        """

        return self.__atom_types

    @property
    def atom_counts(self):
        """
        Retrieves the number of atoms of all types.

        Returns
        -------
        list(int)
            Numbers of atoms.
        """

        return [self.__atom_lists[type_index].shape[0] \
            for type_index in range(self.__atom_types)]

    def atom_count(self, type_specifier):
        """
        Retrieves the number of atoms of a given type.

        Parameters
        ----------
        type_specifier : int or str
            The index or name of the atom type.

        Returns
        -------
        int
            Number of atoms.
        """

        if not isinstance(type_specifier, int):
            type_specifier = self.index(type_specifier)
        return self.__atom_lists[type_specifier].shape[0]

    @property
    def atom_lists(self):
        """
        Retrieves the coordinates of all atoms.

        Returns
        -------
        list(numpy.ndarray)
            List of matrices of row vectors.
        """

        return [self.__atom_lists[type_index].copy() for type_index in range(self.__atom_types)]

    def atoms(self, type_specifier):
        """
        Retrieves the coordinates of atoms of a given type.

        Parameters
        ----------
        type_specifier : int or str
            The index or name of the atom type.

        Returns
        -------
        numpy.ndarray
            Matrix of row vectors containing the coordinates.
        """

        if not isinstance(type_specifier, int):
            type_specifier = self.index(type_specifier)
        return self.__atom_lists[type_specifier].copy()

    def atom(self, type_specifier, atom_index):
        """
        Retrieves the coordinates of a given atom.

        Parameters
        ----------
        type_specifier : int or str
            The index or name of the atom type.
        atom_index : int
            The index of the atom.

        Returns
        -------
        ndarray
            Row vector of the atomic coordinates.
        """

        if not isinstance(type_specifier, int):
            type_specifier = self.index(type_specifier)
        return self.__atom_lists[type_specifier][atom_index].copy()

    @property
    def names(self):
        """
        Retrieves the names of all atom types.

        Returns
        -------
        list(str)
            Names of atom types.
        """

        return list(self.__names)

    def name(self, type_index):
        """
        Retrieves the name of an atom type.

        Parameters
        ----------
        type_index : int
            The index of the atom type.

        Returns
        -------
        str
            Name of the atom type.
        """

        return self.__names[type_index]

    def index(self, type_name):
        """
        Retrieves the index of an atom type.

        Parameters
        ----------
        type_name : str
            The name of the atom type.

        Returns
        -------
        int
            Index of the atom type.
        """

        return self.__names.index(type_name)

    def __scan_shell(self, source_type_index, target_type_index, shell):
        """
        Scans a specific shell of periodic images for contacts between a
        certain pair of atom types.

        Parameters
        ----------
        source_type_index : int
            The index of the source atom type.
        target_type_index : int
            The index of the target atom type.
        shell : int
            Which shell of periodic images to scan.  Specifying 0 scans
            within the cell itself, specifying 1 scans the immediate neighbors
            (8 in 2D, 26 in 3D) around the cell, specifying 2 scans their
            neighbors (16 in 2D, 98 in 3D), and so on.

        Returns
        -------
        collections.Counter(float)
            The generated un-normalized RDF as a :py:class:`collections.Counter`
            object representing a sum of Dirac delta functions.
        """

        # Initialize
        rdf = collections.Counter()
        source_list = self.__atom_lists[source_type_index]
        target_list = self.__atom_lists[target_type_index]

        for cell_coordinates in itertools.product(*(range(-shell, shell + 1) \
            for dimension_index in range(self.__dimensions))):

            # Ignore cell images not in the desired shell
            if numpy.max(numpy.abs(cell_coordinates)) != shell:
                continue

            # Perform the calculations for the current cell image
            shift_vector = numpy.dot(self.__vectors.T, cell_coordinates)
            for source_atom in source_list:
                for target_atom in target_list:
                    distance = numpy.linalg.norm(shift_vector + target_atom - source_atom)
                    if distance:
                        rdf[distance] += 1

        return rdf

    def measure_to(self, *, shell_cutoff=None, distance_cutoff=None):
        """
        Scans pairwise separation distances for all pairs of atoms of all types.
        The measurement can be specified to stop after scanning a certain explicitly
        specified number of periodic image shells, or after scanning enough shells to
        take measurements to a certain distance.  At least one of the two cutoffs must
        be specified; if both are specified, scanning will stop whenever the first
        cutoff is reached.  The generated RDFs will be stored internally within the
        :py:class:`Cell` instance.  Although this method will be invoked automatically
        when contact information or an RDF is requested, it can be called
        manually to generate data that can be saved using :py:func:`write_rdf`.

        Parameters
        ----------
        shell_cutoff : int or None
            Cutoff in terms of periodic image shells
        distance_cutoff : float or None
            Cutoff in terms of pairwise separation distance

        Raises
        ------
        ValueError
            Neither cutoff was specified

        Notes
        -----
        Measurement may be quite slow for large values of the distance cutoff
        parameter and/or cells with a large distortion factor.  A calculation is
        performed to ensure that enough periodic image shells will always be scanned
        to generate accurate information; in some cases, this can be a large number of
        shells.  Consider using :py:func:`CellTools.reduce` to reduce distortion
        and improve performance if necessary.
        """

        if shell_cutoff is None and distance_cutoff is None:
            raise ValueError("one of the two cutoffs must be specified")

        # Determine number of shells to scan
        if distance_cutoff is not None:
            minimum_distance = min([abs(numpy.dot(plane_normal, self.__vectors[index]) / numpy.linalg.norm(plane_normal)) \
                for index, plane_normal in enumerate(self.normals)])
            shell_stop = int(numpy.ceil(distance_cutoff / minimum_distance))
            if shell_cutoff is not None:
                shell_stop = min(shell_stop, shell_cutoff)
        else:
            shell_stop = shell_cutoff

        # Scan shells
        for self.__shells_scanned in range(self.__shells_scanned + 1, shell_stop + 1):
            for source_type_index in range(self.__atom_types):
                for target_type_index in range(source_type_index + 1):
                    self.__rdfs[source_type_index][target_type_index] += \
                        self.__scan_shell(source_type_index, target_type_index, self.__shells_scanned)

    def read_rdf(self, binary_file):
        """
        Reads RDF information from a binary file.

        Parameters
        ----------
        binary_file : file
            The file from which to read information.

        Notes
        -----
        Reading RDF information from a cell with different vectors and atomic
        coordinates may not result in immediate errors if the dimensions and numbers
        of atoms in the cells are equal; however, RDFs and contact information may
        be invalid.
        """

        self.__rdfs, self.__shells_scanned = \
            pickle.load(binary_file)

    def write_rdf(self, binary_file):
        """
        Writes RDF information to a text file.

        Parameters
        ----------
        binary_file : file
            The file to which to write information.
        """

        pickle.dump((self.__rdfs, self.__shells_scanned), \
            binary_file)

    def rdf(self, source_type_specifier, target_type_specifier, distance):
        """
        Retrieves a discrete RDF.  No normalization is applied.
        This calculation is simply a sum or total counts each time two
        particles are found a certain distance apart.

        Parameters
        ----------
        source_type_specifier : int or str
            The index or name of the source atom type.
        target_type_specifier : int or str
            The index or name of the target atom type.
        distance : float
            The distance to which the RDF should be measured.

        Returns
        -------
        dict(float, int)
            RDF as a dictionary.  Each entry has as its key a distance, and
            as its value the number of atoms of the target type found at a
            distance away from atoms of the source type.
        """

        if not isinstance(source_type_specifier, int):
            source_type_specifier = self.index(source_type_specifier)
        if not isinstance(target_type_specifier, int):
            target_type_specifier = self.index(target_type_specifier)
        if target_type_specifier > source_type_specifier:
            source_type_specifier, target_type_specifier = target_type_specifier, source_type_specifier

        self.measure_to(distance_cutoff=distance)
        return { key: value for key, value in self.__rdfs[source_type_specifier][target_type_specifier].items() if key <= distance }

    def contact(self, source_type_specifier, target_type_specifier):
        """
        Retrieves atomic minimum contact information.

        Parameters
        ----------
        source_type_specifier : int or str
            The index or name of the source atom type.
        target_type_specifier : int or str
            The index or name of the target atom type.

        Returns
        -------
        float
            The minimum distance between atoms of the two types.
        """

        if not isinstance(source_type_specifier, int):
            source_type_specifier = self.index(source_type_specifier)
        if not isinstance(target_type_specifier, int):
            target_type_specifier = self.index(target_type_specifier)
        if target_type_specifier > source_type_specifier:
            source_type_specifier, target_type_specifier = target_type_specifier, source_type_specifier

        return __fast_contact_atoms(
            self.__atom_lists[source_type_specifier],
            self.__atom_lists[target_type_specifier],
            self.__vectors,
            source_type_specifier == target_type_specifier)

    def scale_factor(self, radii):
        """
        Calculates (based on minimum contact information) a scale factor between
        the length scale of provided atomic radii and the length scale of cell
        vectors and atomic positions.

        Parameters
        ----------
        radii : tuple(float)
            The radii of the atom types.

        Raises
        ------
        ValueError
            An invalid number of radii were provided.

        Returns
        -------
        float
            The requested scale factor.

        Notes
        -----
        Suppose that the cell vectors' coordinates have physically significant units
        and the atomic radii provided are relative.  In this case, multiply the radii
        by the scale factor to obtain their values in the units of the cell vectors.

        Alternatively, suppose that the atomic radii provided are in physically
        significant units and the cell vectors' coordinates are relative.  Dividing
        the coordinates of the cell vectors by the scale factor will yield their values
        in the units of the atomic radii.

        **All particles should be wrapped inside the periodic box before this routine is
        called.**  The routine will not function as expected otherwise.
        """

        # Check number of radii
        if len(radii) != len(self.__atom_lists):
            raise ValueError("Number of radii must correspond with number of atom types")

        # Rescale radii for proper cutoff information
        ratio = lambda indices: self.contact(*indices) / sum(radii[index] for index in indices)
        source_type_index, target_type_index = min(((source_type_index, target_type_index) \
            for source_type_index in range(len(self.__atom_lists)) \
            for target_type_index in range(len(self.__atom_lists))), key=ratio)
        return ratio((source_type_index, target_type_index))

class CellTools:
    """
    Contains tools for performing simple manipulations on cells, such as scaling and shearing cells,
    generating periodic supercells, wrapping atoms in accordance with periodic boundary conditions,
    optimization of sheared cell representations, and energy calculation.
    """

    @staticmethod
    def similar(cell_1, cell_2, tolerance=1e-6):
        """
        Tests whether or not two cells are equal.  The comparison of numerical values
        between the two cells is performed with some amount of tolerance.
        This is **NOT** a similarity metric, as in :py:class:`paccs.similarity.SimilarityMetric`.

        Parameters
        ----------
        cell_1 : Cell
            The first cell to compare.
        cell_2 : Cell
            The second cell to compare.
        tolerance : float
            The tolerance to use during comparison of cell vectors and atom
            coordinates.

        Returns
        -------
        bool
            Whether or not the cells are equal.
        """

        return cell_1.dimensions == cell_2.dimensions and cell_1.atom_counts == cell_2.atom_counts and \
            numpy.allclose(cell_1.vectors, cell_2.vectors, rtol=tolerance, atol=tolerance) and \
            all(numpy.allclose(cell_1.atoms(index), cell_2.atoms(index), rtol=tolerance, atol=tolerance) \
            for index in range(cell_1.atom_types)) and cell_1.names == cell_2.names

    @staticmethod
    def identical(cell_1, cell_2):
        """
        Tests whether or not two cells are equal.  The comparison of numerical values
        between the two cells is performed exactly.

        Parameters
        ----------
        cell_1 : Cell
            The first cell to compare.
        cell_2 : Cell
            The second cell to compare.

        Returns
        -------
        bool
            Whether or not the cells are equal.
        """

        return cell_1.dimensions == cell_2.dimensions and cell_1.atom_counts == cell_2.atom_counts and \
            numpy.array_equal(cell_1.vectors, cell_2.vectors) and \
            all(numpy.array_equal(cell_1.atoms(index), cell_2.atoms(index)) \
            for index in range(cell_1.atom_types)) and cell_1.names == cell_2.names

    @staticmethod
    def rename(cell, type_map):
        """
        Renames atom types in a cell.  The manner in which the atoms are stored internally will
        not be modified, only the names of the atom types will be reassigned.

        Parameters
        ----------
        cell : Cell
            The cell to modify.
        type_map : dict(int or str, int or str)
            A dictionary mapping current atom type identifiers to new atom type identifiers.
            Any indices will be converted to their respective names before renaming.

        Raises
        ------
        TypeError
            Invalid or inconsistent types in type map.
        ValueError
            Unrecognized indices or names, or invalid type map specification.

        Returns
        -------
        Cell
            The modified cell.
        """

        # Converts indices to names
        def normalize(item):
            if isinstance(item, int):
                return cell.name(item)
            return item

        # Check types in type map
        normalized_type_map = {}
        for key, value in type_map.items():
            normalized_type_map[normalize(key)] = normalize(value)
        type_map = normalized_type_map

        # Try to perform the renaming operations
        new_names = cell.names
        for key, value in type_map.items():
            if key not in cell.names:
                raise ValueError("invalid value encountered in type map")
            new_names[cell.index(key)] = value

        # Check result
        if len(set(new_names)) != len(new_names):
            raise ValueError("invalid value encountered in type map")

        return Cell(cell.vectors, cell.atom_lists, new_names)


    @staticmethod
    def reassign(cell, type_map):
        """
        Reassigns or reorders atom types in a cell.  Multiple atom types can be merged
        into one, and atom types can be dropped completely if desired.

        Parameters
        ----------
        cell : Cell
            The cell to modify.
        type_map : dict(int or str, int or str)
            A dictionary mapping current atom type identifiers to new atom type identifiers.
            Any names will be converted to their respective indices before renaming.

        Raises
        ------
        TypeError
            Invalid or inconsistent types in type map.
        ValueError
            Unrecognized indices or names, or invalid type map specification.

        Returns
        -------
        Cell
            The modified cell.
        """

        # Converts names to indices
        def normalize(item):
            if not isinstance(item, int):
                return cell.index(item)
            return item

        # Check types in type map
        normalized_type_map = {}
        for key, value in type_map.items():
            normalized_type_map[normalize(key)] = normalize(value)
        type_map = normalized_type_map

        # Check indices for validity
        sources, destinations = set(type_map.keys()), set(type_map.values())
        if len(sources) > cell.atom_types or len(destinations) != max(destinations) + 1 \
            or any(source not in range(cell.atom_types) for source in sources) \
            or min(destinations) != 0:
                raise ValueError("invalid entry encountered in type map")

        # Reorder atoms, possibly with condensation
        atom_lists = [numpy.array([cell.atom(source_type_index, atom_index) \
            for source_type_index in range(max(sources) + 1) \
            for atom_index in range(cell.atom_count(source_type_index)) \
            if source_type_index in type_map and type_map[source_type_index] == target_type_index]) \
            for target_type_index in range(max(destinations) + 1)]

        # Generate names, possibly losing names if atom types are being lost
        names = [cell.name(min(source_type_index \
            for source_type_index in range(max(sources) + 1) \
            if source_type_index in type_map and type_map[source_type_index] == target_type_index)) \
            for target_type_index in range(max(destinations) + 1)]

        return Cell(cell.vectors, atom_lists, names)

    @staticmethod
    def scale(cell, new_vectors, move_vectors=True, move_atoms=True):
        """
        Rescales a cell by applying new vectors.

        Parameters
        ----------
        cell : Cell
            The cell to scale.
        new_vectors : numpy.ndarray
            A matrix containing new row vectors.
        move_vectors : bool
            Whether or not to apply the scaling to the vectors of the cell.
        move_atoms : bool
            Whether or not to apply the scaling to the atoms of the cell.

        Raises
        ------
        ValueError
            Invalid vectors were received.

        Returns
        -------
        Cell
            The modified cell.
        """

        # Check the vectors
        vectors = cell.vectors
        if new_vectors.shape != vectors.shape:
            raise ValueError("vectors must be provided as a square matrix")

        # Generate transformation matrices from old coordinate space to normalized space
        # (in which all coordinates inside the cell are in the range [0, 1]) and back to
        # the new coordinate space with new vectors
        normalized_to_coordinate = new_vectors.T
        coordinate_to_normalized = scipy.linalg.inv(vectors.T)

        # Adjust the coordinates
        atom_lists = [numpy.dot(normalized_to_coordinate, numpy.dot(coordinate_to_normalized, cell.atoms(type_index).T)).T \
            for type_index in range(cell.atom_types)] \
            if move_atoms else cell.atom_lists

        # Adjust the vectors
        return Cell(new_vectors if move_vectors else vectors, atom_lists, cell.names)

    @staticmethod
    def normalize(cell):
        """
        Normalizes the vectors of a cell such that the vector matrix is triangular.
        The atoms within are moved to compensate such that histograms should not be
        changed. This is the format for triclinic cells used by LAMMPS.

        Parameters
        ----------
        cell : Cell
            The cell whose vectors are to be normalized.

        Returns
        -------
        Cell
            The modified cell.
        """

        # Retrieve cell vectors and create array for new vectors
        vectors = cell.vectors
        new_vectors = numpy.zeros_like(vectors)

        # Before beginning, check sign of determinant and flip a vector if needed
        if numpy.linalg.det(vectors) < 0:
            vectors[0] *= -1
            cell = CellTools.wrap(Cell(vectors, cell.atom_lists))

        # Populate matrix
        for component_index in range(cell.dimensions):
            # Fill diagonal element of this column
            new_vectors[component_index][component_index] = \
                numpy.sqrt((numpy.linalg.norm(vectors[component_index]) ** 2) - \
                sum(new_vectors[component_index][index] ** 2 \
                for index in range(component_index)))

            # Fill off diagonal (lower triangular) elements
            for vector_index in range(component_index, cell.dimensions):
                new_vectors[vector_index][component_index] = \
                    (numpy.dot(vectors[component_index], vectors[vector_index]) - \
                    sum(new_vectors[component_index][index] * new_vectors[vector_index][index] \
                    for index in range(component_index))) / new_vectors[component_index][component_index]

        return CellTools.scale(cell, new_vectors)

    @staticmethod
    def wrap(cell):
        """
        Wraps atoms within a cell based on periodic boundary conditions.

        Parameters
        ----------
        cell : Cell
            The cell whose atoms are to be wrapped.

        Returns
        -------
        Cell
            The modified cell.
        """

        # Generate transformation matrices from coordinate space to normalized space
        # (in which all coordinates inside the cell are in the range [0, 1]) and back
        normalized_to_coordinate = cell.vectors.T
        coordinate_to_normalized = scipy.linalg.inv(normalized_to_coordinate)

        # Transform, wrap, and transform back
        atom_lists = [numpy.dot(normalized_to_coordinate, (numpy.dot(coordinate_to_normalized, cell.atoms(type_index).T) % 1)).T \
            for type_index in range(cell.atom_types)]
        return Cell(cell.vectors, atom_lists, cell.names)

    @staticmethod
    def condense(cell, wrap=True, tolerance=1e-6):
        """
        Removes duplicate atoms from a cell.  Only atoms of the same type are
        candidates for being duplicates.  This can be useful, for instance,
        to extract a unit cell from periodic cell data using::

            cell = CellTools.condense(CellTools.scale(cell, new_vectors, move_atoms=False))

        This routine only considers nearest neighbor cells, i.e., only
        +/- 1 translation by the cell's vectors.

        Parameters
        ----------
        cell : Cell
            The cell from which to remove duplicate atoms.
        wrap : bool
            Whether or not to apply :py:func:`wrap` before condensing (if atoms exist
            outside the cell, duplicate detection may function improperly).
        tolerance : float
            The distance two atoms of the same type must be from each other
            to be considered separate atoms.

        Returns
        -------
        Cell
            The modified cell.
        """

        # Wrap first if desired
        if wrap:
            cell = CellTools.wrap(cell)

        atom_lists = []
        for type_index in range(cell.atom_types):
            # Call optimized routine to identify overlapping atoms subject to periodic
            # boundary conditions with the triclinic box
            overlaps = __fast_squareform_pdist_overlap(cell.atoms(type_index), cell.vectors, tolerance)

            # Look at all pairs of atoms
            atoms_to_delete = set()
            for source_atom_index in range(cell.atom_count(type_index)):
                for target_atom_index in range(cell.atom_count(type_index)):
                    # Don't process pairs with the same atom selected twice
                    if source_atom_index == target_atom_index:
                        continue

                    # If an overlap is found, remove one atom and update the array
                    if overlaps[source_atom_index][target_atom_index]:
                        atoms_to_delete.add(target_atom_index)
                        overlaps[source_atom_index, target_atom_index] = 0
                        overlaps[target_atom_index, :] = 0

            # Rebuild the atom list for this atom type
            atom_lists.append(numpy.array([cell.atom(type_index, atom_index) \
                for atom_index in range(cell.atom_count(type_index)) \
                if atom_index not in atoms_to_delete]))
        return Cell(cell.vectors, atom_lists, cell.names)

    @staticmethod
    def shift(cell, atom_type=0, atom_index=0, position=None):
        """
        Shifts atoms in a cell such that the specified atom is at a
        desired position.

        Parameters
        ----------
        cell : Cell
            The cell to modify.
        atom_type : int or str
            The type (index or name) of the atom to select as a reference
            point.  If not specified, defaults to the first type.
        atom_index : int
            The index of the atom (of the selected type) to choose as
            a reference point.  If not specified, defaults to the first
            atom of the selected type.
        position : numpy.ndarray
            A row vector specifying the desired new position of the
            reference atom.  All atoms in the cell will be shifted by
            the same vector to effect this change.  If unspecified, the
            selected atom will be moved to the origin.

        Returns
        -------
        Cell
            The modified cell.
        """

        # Check arguments
        if isinstance(atom_type, str):
            atom_type = cell.index(atom_type)
        if position is None:
            position = numpy.zeros((1, cell.dimensions))

        # Determine required vector and shift the atoms
        shift_vector = position - cell.atom(atom_type, atom_index)
        return Cell(cell.vectors, [cell.atoms(type_index) + \
            numpy.tile(shift_vector, cell.atom_count(type_index)).reshape(-1, cell.dimensions) \
            for type_index in range(cell.atom_types)], cell.names)

    @staticmethod
    def tile(cell, repeats, partial_radii=None, partial_tolerance=1e-6):
        """
        Generates a periodic supercell from the primitive cell.  If desired, additional atoms can
        be added at the edges of the resulting cell (for example, when visualizing).
        The supercell is scaled to closest contact based on the radii, if specified.

        Parameters
        ----------
        cell : Cell
            The primitive cell to tile using the cell's vectors.
        repeats : tuple(int)
            The number of repeats to make in each direction.
        partial_radii : tuple(float)
            The radii of atoms of each type in the cell, if it is desired to
            add additional atoms.  These values are used to determine whether
            an atom whose center lies outside the final supercell is still
            partially within its bounds and should be included.  If this parameter
            is not specified, no additional atoms will be added to the supercell.
        partial_tolerance : float
            An adjustment value used when determining whether an atom whose center
            is outside of a cell still rests within it.  The distance from the
            center of a sphere to the plane cutting through it must be this much
            less than the radius of the sphere.

        Raises
        ------
        ValueError
            Inconsistent dimensions or invalid specifications.

        Returns
        -------
        Cell
            The generated supercell.
        """

        # Check the repeat counts
        if len(repeats) != cell.dimensions:
            raise ValueError("repeats must be consistent with cell dimensions")
        for repeat in repeats:
            if repeat != int(repeat) or repeat < 1:
                raise ValueError("repeat values must be positive integers")

        # Add extra repeats at beginning and end if partials are requested
        repeats = numpy.array(repeats)
        new_vectors = cell.vectors * numpy.repeat(repeats, cell.dimensions).reshape(cell.dimensions, cell.dimensions)
        if partial_radii is not None:
            transformation = scipy.linalg.inv(new_vectors.T)
            repeats += 2

            # Rescale radii for proper cutoff information
            partial_radii = numpy.array(partial_radii) * cell.scale_factor(partial_radii)

            # Precompute plane normals
            precomputed_normals = cell.normals

            # Determines the nearest distance from a plane to a point
            plane_point = lambda plane_origin, plane_normal, target_point: \
                abs(numpy.dot(plane_normal, target_point - plane_origin) / numpy.linalg.norm(plane_normal))

            # Define criterion for retaining atom
            def good_partial(type_index, atom_coordinates):
                normalized_coordinates = numpy.dot(transformation, atom_coordinates.T).T
                under_range, over_range = normalized_coordinates < 0, normalized_coordinates > 1
                if numpy.any(under_range) or numpy.any(over_range):
                    # Check distance to all planes in question
                    for dimension_index in range(cell.dimensions):
                        if under_range[dimension_index]:
                            if plane_point(numpy.zeros(cell.dimensions), precomputed_normals[dimension_index], \
                                atom_coordinates) >= partial_radii[type_index] - partial_tolerance:
                                return False
                    for dimension_index in range(cell.dimensions):
                        if over_range[dimension_index]:
                            if plane_point(new_vectors[dimension_index], precomputed_normals[dimension_index], \
                                atom_coordinates) >= partial_radii[type_index] - partial_tolerance:
                                return False
                    # Atom was in contact with all planes for which it was out of range
                    return True
                else:
                    # Atom is most definitely within cell
                    return True

        # Generate new cell vectors and atom lists
        vectors = cell.vectors.T
        atom_lists = [numpy.array([coordinates + numpy.dot(vectors, (numpy.array(offset) - (1 if partial_radii is not None else 0))) \
            for coordinates in cell.atoms(type_index) \
            for offset in itertools.product(*(range(repeat) \
            for repeat in repeats)) \
            if partial_radii is None or good_partial(type_index, coordinates + numpy.dot(vectors, (numpy.array(offset) - (1 if partial_radii is not None else 0))))]) \
            for type_index in range(cell.atom_types)]

        # Remove repeats
        if partial_radii is not None:
            repeats -= 2

        return Cell(new_vectors, atom_lists, cell.names)

    @staticmethod
    def reduce(cell, max_distortion=1.5, max_iterations=10, normalize=True, shift=True, wrap=True, condense=True, tolerance=1e-6):
        """
        Reduces a sheared cell based on the algorithm discussed in *de Graaf et al.*
        (`doi:10.1063/1.4767529 <https://dx.doi.org/10.1063/1.4767529>`_).  This is
        useful to prevent energy calculations from becoming excessively costly as a
        cell shears during optimization.

        Parameters
        ----------
        cell : Cell
            The cell to reduce.
        max_distortion : float
            The maximum permissible distortion factor.  This value is equal to 1 for
            a non-sheared cell.
        max_iterations : int
            The maximum number of iterations that will be executed.  Attempts to reduce
            the cell will stop when either the distortion factor decreases below the
            specified threshold or the number of iterations is exceeded.
        normalize : bool
            Whether or not to normalize cell vectors and atom positions such that the
            cell vector matrix is triangular.
        shift : bool
            Whether or not to shift atoms within the cell such that the first atom of
            the first type is at the origin.
        wrap : bool
            Whether or not to clean up the cell using :py:func:`wrap` after optimization.
        condense : bool
            Whether or not to clean up the cell using :py:func:`condense` after optimization.
        tolerance : float
            The tolerance used when calling :py:func:`condense`.

        Returns
        -------
        Cell
            The modified cell.
        """

        # Prepare a special cell containing vectors alone to avoid overhead of
        # copying around atom position arrays
        vector_cell = Cell(cell.vectors, [], [])

        # Perform at maximum the specified number of iterations
        for iteration in range(max_iterations):
            # If the cell is not particularly distorted at this point, stop
            if cell.distortion_factor <= max_distortion:
                break

            # Calculate the initial surface space and look for a reduction
            initial_surface = vector_cell.surface
            new_cells = [Cell(numpy.array([ \
                    vector_cell.vector(vector_index) + (direction * vector_cell.vectors[modifying_vector_index] \
                    if modified_vector_index == vector_index else 0) \
                    for vector_index in range(cell.dimensions)
                ]), [], []) \
                for direction in [-1, 1]
                for modified_vector_index in range(cell.dimensions) \
                for modifying_vector_index in range(cell.dimensions) \
                if modified_vector_index != modifying_vector_index]

            # Find the best modified cell
            old_vector_cell = vector_cell
            vector_cell = min(new_cells + [vector_cell], key=lambda cell: cell.surface)
            # The original cell was selected, no further iterations will improve outcome
            if vector_cell is old_vector_cell:
                break

        # Perform cleaning operations if desired and deliver the result
        cell = Cell(vector_cell.vectors, cell.atom_lists, cell.names)
        if shift:
            cell = CellTools.shift(cell)
        if normalize:
            cell = CellTools.normalize(cell)
        if wrap:
            cell = CellTools.wrap(cell)
        if condense:
            cell = CellTools.condense(cell, tolerance)
        return cell

    @staticmethod
    def energy(cell, potentials, distance):
        """
        Determines the interaction energy per atom of a cell by evaluating
        pair potentials out to a certain distance.

        Parameters
        ----------
        cell : Cell
            The cell whose energy is to be evaluated.
        potentials: dict(tuple(int or str), paccs.potential.Potential)
            Potentials for each pair of atom types.  If a potential is not specified
            for a given pair, then no interactions will occur between those atoms.
        distance : float
            Distance to which evaluation should take place.

        Raises
        ------
        TypeError
            Invalid atom type specifier type.
        ValueError
            Invalid specification of atom type pairs was encountered.

        Returns
        -------
        float
            Interaction energy per atom.

        Notes
        -----
        **It is generally recommended that atom names (as strings, instead of ints)
        be used as pair types in the potentials dictionary to avoid ambiguity.**
        """

        # Generate a potential array
        zero_potential = lambda r: 0.0
        potential_array = [[zero_potential] * cell.atom_types for type_index in range(cell.atom_types)]
        for pair in potentials:
            source_type_index, target_type_index = pair

            # Convert names to indices
            try:
                if not isinstance(source_type_index, int):
                    source_type_index = cell.index(source_type_index)
                if not isinstance(target_type_index, int):
                    target_type_index = cell.index(target_type_index)
            except:
                # If either of these names/identifiers aren't in the cell, skip
                continue

            # Check if these types in the potential dictionary are present in the cell
            for tt in [source_type_index, target_type_index]:
                if (tt < 0 or tt >= cell.atom_types):
                    raise Exception('illegal type index {} for a cell with only {} atom types - specify potentials using atom string names if you wish to provide potential information for cells missing certain atom types'.format(tt, cell.atom_types))

            # Do the potential assignment
            if potential_array[source_type_index][target_type_index] is not zero_potential:
                raise ValueError("duplicate pair potential assignment encountered")
            potential_array[source_type_index][target_type_index] = potentials[pair]
            if source_type_index != target_type_index:
                potential_array[target_type_index][source_type_index] = potentials[pair]

        # Determine the potential energies
        energy = 0.0
        for source_type_index in range(cell.atom_types):
            for target_type_index in range(cell.atom_types):
                energy += 0.5 * sum(value * potential_array[source_type_index][target_type_index].evaluate(key)[0] \
                    for key, value in cell.rdf(source_type_index, target_type_index, distance).items())
        return energy / sum(cell.atom_counts)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef __fast_squareform_pdist_overlap(
        numpy.ndarray[numpy.float64_t, ndim=2] atoms,
        numpy.ndarray[numpy.float64_t, ndim=2] vectors,
        double tolerance):

    # This routine:
    #     overlaps = __fast_squareform_pdist_overlap(atoms, vectors, tolerance)
    # is an optimized replacement for the following:
    #     overlaps = scipy.spatial.distance.squareform(
    #      scipy.spatial.distance.pdist(
    #       atoms, metric=lambda u, v:
    #        min(numpy.linalg.norm(u - v + numpy.dot(vectors.T, numpy.array(offset)))
    #         for offset in itertools.product(*(range(-1, 2)
    #          for dimension_index in range(atoms.shape[1])))))) < tolerance

    # Precompute offsets (this is a fast equivalent of the itertools.product invocation)
    cdef long dimension_count = atoms.shape[1]
    cdef numpy.ndarray[numpy.int64_t, ndim=2] offsets = numpy.array(numpy.rollaxis( \
        numpy.indices([3] * dimension_count) - 1, 0, dimension_count + 1).reshape(-1, dimension_count), dtype=numpy.int64)
    cdef long offset_count = offsets.shape[0]
    cdef long offset_index
    cdef numpy.ndarray[numpy.float64_t, ndim=2] offset_vectors = numpy.zeros_like(offsets, dtype=numpy.float64)
    for offset_index in range(offset_count):
        offset_vectors[offset_index, :] = numpy.dot(vectors.T, offsets[offset_index, :])

    # Prepare output array for overlap results (use uint8_t to represent bool)
    cdef long atom_count = atoms.shape[0]
    cdef numpy.ndarray[numpy.uint8_t, ndim=2] result = numpy.zeros((atom_count, atom_count), dtype=numpy.uint8)

    cdef long source_atom_index, target_atom_index, dimension_index
    cdef double minimum_distance, current_distance, current_term
    cdef double inf = numpy.finfo(numpy.float64).max

    # Loop over all pairs of atoms, noting that the metric is symmetric
    # with respect to swapping indices (i, j) <--> (j, i)
    for source_atom_index in range(atom_count):
        for target_atom_index in range(source_atom_index, atom_count):
            # We are looking for the minimum distance in all periodic images
            minimum_distance = inf
            for offset_index in range(offset_count):
                # Compute norm(atom_i - atom_j + (vectors.T dot image_offset))
                current_distance = 0
                for dimension_index in range(dimension_count):
                    current_term = atoms[source_atom_index, dimension_index] \
                        - atoms[target_atom_index, dimension_index] \
                        + offset_vectors[offset_index, dimension_index]
                    current_distance += current_term * current_term
                if current_distance < minimum_distance:
                    minimum_distance = current_distance
            # At this point, we have the minimum (squared) distance between atoms
            if minimum_distance < tolerance * tolerance:
                result[source_atom_index, target_atom_index] = 1
                result[target_atom_index, source_atom_index] = 1

    return result

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef __fast_contact_atoms(
        numpy.ndarray[numpy.float64_t, ndim=2] source_atoms,
        numpy.ndarray[numpy.float64_t, ndim=2] target_atoms,
        numpy.ndarray[numpy.float64_t, ndim=2] vectors,
        char same_types):

    # This routine:
    #     __fast_contact_atoms(cell.__atom_lists[source_type_specifier],
    #         cell.__atom_lists[target_type_specifier], cell.__vectors,
    #         source_type_specifier == target_type_specifier)
    # is an optimized replacement for the following:
    #     cell.measure_to(shell_cutoff=1)
    #     min(cell.__rdfs[source_type_specifier], cell.__rdfs[target_type_specifier])
    # This is used in places where accurate contact information is needed rapidly.
    # For instance, scale_factor should execute quickly without having to call
    # __scan_shell and create an inefficient collections.Counter object.

    # Precompute offsets (this is a fast equivalent of the itertools.product invocation)
    cdef long dimension_count = source_atoms.shape[1]
    cdef long dimension_index
    cdef numpy.ndarray[numpy.int64_t, ndim=2] offsets = numpy.array(numpy.rollaxis( \
        numpy.indices([3] * dimension_count) - 1, 0, dimension_count + 1).reshape(-1, dimension_count), dtype=numpy.int64)
    cdef long offset_count = offsets.shape[0]
    cdef long offset_index
    cdef numpy.ndarray[numpy.float64_t, ndim=2] offset_vectors = numpy.zeros_like(offsets, dtype=numpy.float64)
    for offset_index in range(offset_count):
        offset_vectors[offset_index, :] = numpy.dot(vectors.T, offsets[offset_index, :])

    # Loop over all pairs of atoms: unlike __fast_squareform_pdist_overlap, the source
    # and target atom lists might be different so the assumption (i, j) <=> (j, i) does
    # not hold in general.
    cdef long source_atom_count = source_atoms.shape[0]
    cdef long target_atom_count = target_atoms.shape[0]
    cdef long source_atom_index, target_atom_index
    cdef double minimum_distance = numpy.finfo(numpy.float64).max
    cdef double current_distance, current_term
    for source_atom_index in range(source_atom_count):
        for target_atom_index in range(source_atom_index if same_types else 0, target_atom_count):
            for offset_index in range(offset_count):
                # Skip this case if we are looking at the exact same atom in its own periodic image
                # The (0, 0, ...) offset index will be directly in the center of the offset array
                if same_types and source_atom_index == target_atom_index and offset_index * 2 == offset_count - 1:
                    continue
                # We are looking for the minimum distance for *any* atom pair in *any* periodc image
                current_distance = 0
                for dimension_index in range(dimension_count):
                    current_term = source_atoms[source_atom_index, dimension_index] \
                        - target_atoms[target_atom_index, dimension_index] \
                        + offset_vectors[offset_index, dimension_index]
                    current_distance += current_term * current_term
                if current_distance < minimum_distance:
                    minimum_distance = current_distance

    assert minimum_distance != numpy.finfo(numpy.float64).max
    return libc.math.sqrt(minimum_distance)

class CellCodecs:
    """
    Contains various codecs for loading and saving :py:class:`Cell` objects' data in different formats.
    """

    @staticmethod
    def read_cell(text_file):
        """
        Reads cell data from an ASCII formatted ".cell" file.  The format is documented
        with the :py:func:`write_cell` function.

        Parameters
        ----------
        text_file : file
            The input file.

        Raises
        ------
        ValueError
            Invalid data were encountered.

        Returns
        -------
        Cell
            The cell created from the file.
        """

        lines = (line for line in (line.strip() for line in text_file) if line)

        # Read the heading line
        lengths = [int(value.strip()) for value in next(lines).split()]
        dimensions, list_lengths = lengths[0], lengths[1:]
        if dimensions < 0:
            raise ValueError("number of dimensions must not be negative")
        for list_length in list_lengths:
            if list_length < 0:
                raise ValueError("number of atoms must not be negative")

        # Read atom names
        names = [None] * len(list_lengths)
        for name_index in range(len(list_lengths)):
            names[name_index] = next(lines).encode().decode("unicode_escape")

        # Read vector data
        vectors = [None] * dimensions
        for vector_index in range(dimensions):
            vector = [float(value.strip()) for value in next(lines).split()]
            if len(vector) != dimensions:
                raise ValueError("vector coordinates must be consistent with cell dimensions")
            vectors[vector_index] = vector

        # Read atom data
        atom_lists = [None] * len(list_lengths)
        for type_index in range(len(list_lengths)):
            atom_list = [None] * list_lengths[type_index]
            for atom_index in range(list_lengths[type_index]):
                coordinates = [float(value.strip()) for value in next(lines).split()]
                if len(coordinates) != dimensions:
                    raise ValueError("atom coordinates must be consistent with cell dimensions")
                atom_list[atom_index] = coordinates
            atom_lists[type_index] = atom_list

        # Ensure that the end of the file has been reached
        try:
            next(lines)
            raise ValueError("unexpected additional data encountered")
        except StopIteration:
            pass

        return Cell(vectors, atom_lists, names)

    @staticmethod
    def write_cell(cell, text_file):
        """
        Writes cell data to an ASCII formatted ".cell" file.  These files consist of lines of
        either text (escaped with backslashes in standard Python format) or numbers (separated
        from each other on a line by whitespace).  Comments are not supported but blank lines
        are.

        The first (non-blank) line is a header line, containing the number of dimensions (almost
        always 2 or 3) followed by the number of atoms of each type.  The number of values following
        the dimensions thus specifies the number of atom types in the cell.  This line is followed
        by a number of lines specifying the names of each atom type.

        This is followed by N lines containing N values each, where N is the number of dimensions.
        These lines specify the cell vectors (in row vector format).  Finally, some number of lines
        with N values each follow.  These lines specify the positions of atoms (also in row vector
        format).  All of the atoms of the first type are positioned, followed by the atoms of the
        second type, and so on.

        Parameters
        ----------
        cell : Cell
            The cell to save.
        text_file : file
            The output file.
        """

        # Write the heading line
        text_file.write("{} {}\n".format(cell.dimensions, " ".join(str(count) for count in cell.atom_counts)))

        # Write atom names
        for name in cell.names:
            text_file.write("{}\n".format(name.encode("unicode_escape").decode()))

        # Write vector data
        for vector_index in range(cell.dimensions):
            text_file.write("{}\n".format(" ".join("{:.16e}".format(coordinate) \
                for coordinate in cell.vector(vector_index))))

        # Write atom data
        for type_index in range(cell.atom_types):
            for atom_index in range(cell.atom_count(type_index)):
                text_file.write("{}\n".format(" ".join("{:.16e}".format(coordinate) \
                    for coordinate in cell.atom(type_index, atom_index))))

    @staticmethod
    def write_xyz(cell, text_file):
        """
        Writes cell data to an XYZ file.  Vector information is placed directly into the
        comment line of the file.

        Parameters
        ----------
        cell : Cell
            The cell to export.
        text_file : file
            The output file.
        """

        # Write header information
        text_file.write("{}\n".format(sum(cell.atom_counts)))
        text_file.write("{}\n".format(" ".join(repr(float(coordinate)) \
            for coordinate in cell.vectors.flatten())))

        # Write atom data
        for type_index in range(cell.atom_types):
            name = cell.name(type_index)
            for atom_index in range(cell.atom_count(type_index)):
                text_file.write("{} {}{}\n".format(name, " ".join(repr(float(coordinate)) \
                    for coordinate in cell.atom(type_index, atom_index)), \
                    " 0.0" if cell.dimensions == 2 else ""))

    @staticmethod
    def write_lammps(cell, text_file):
        """
        Writes cell data to a LAMMPS file.  Atom positions are changed in accordance
        with LAMMPS' normalization scheme for triclinic cells.

        Raises
        ------
        NotImplementedError
            The cell does not have 2 or 3 dimensions

        Parameters
        ----------
        cell : Cell
            The cell to export.
        text_file : file
            The output file.
        """

        # Normalize the cell
        if cell.dimensions not in {2, 3}:
            raise NotImplementedError("LAMMPS export supported in 2D and 3D only")
        cell = CellTools.normalize(cell)

        # Write header information
        text_file.write("LAMMPS\n\n{} atoms\n{} atom types\n\n".format( \
            sum(cell.atom_counts), cell.atom_types))

        # Write box data
        A, B, C = cell.vectors if cell.dimensions == 3 else (tuple(cell.vectors) + (numpy.zeros((2,)),))
        text_file.write("0.0 {!r} xlo xhi\n".format(A[0]))
        text_file.write("0.0 {!r} ylo yhi\n".format(B[1]))
        text_file.write("{!r} {!r} zlo zhi\n".format(*((0.0, C[2]) if cell.dimensions == 3 else (-1.0, 1.0))))
        text_file.write("{!r} {!r} {!r} xy xz yz\n\nAtoms\n\n".format(B[0], C[0], C[1]))

        # Write actual atom positions
        atom_counter = 0
        for type_index in range(cell.atom_types):
            atom_list = cell.atoms(type_index)
            for atom_index in range(cell.atom_count(type_index)):
                text_file.write("{} {} {}{}\n".format( \
                    atom_counter + 1, type_index + 1, " ".join(repr(float(coordinate)) \
                    for coordinate in cell.atom(type_index, atom_index)), \
                    " 0.0" if cell.dimensions == 2 else ""))
                atom_counter += 1
