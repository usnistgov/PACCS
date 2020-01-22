"""
Wallpaper groups for two-dimensional crystal operations.
"""

from . import crystal
from . import enum_config as enumconfig
import fractions
import constraint
import collections
import functools
import itertools
import math
import numpy
import sys
import time
import warnings

# For compact descriptive representation of groups
class _tuple(tuple):
    def __new__(class_, *args):
        return super(_tuple, class_).__new__(class_, tuple(args))

# For compact creation of fractions
_F = fractions.Fraction

def Nx(Ng, group, r=1):
    r"""
    Get the number of grid points to use for a group's fundamental domain,
    given a number for a "congruent" p1 primitive cell. This number is the number needed
    to achieve the maximum nodal density at or below that of the p1 primitive
    cell being used as a benchmark.

    Parameters
    ----------
    Ng : int
        Number of grid points to use on the p1 primitive cell. Should correspond
        to the smallest edge.
    group : str
        Hermann-Mauguin wallpaper group name of interest.
    r : float
        Length ratio to use, and should be >= 1.

    Returns
    -------
    Nx : int
        Number of grid points to use on a groups fundamental domain.
    """

    assert (r >= 1), 'length ratio should be >= 1 - have not considered this case yet'

    N_points = {
        'p1': int(Ng),
        'p2': int(numpy.floor(Ng/numpy.sqrt(2.*r))),
        'p3': int(numpy.floor(Ng/numpy.sqrt(3.*r))),
        'p4': int(numpy.floor(Ng/2./numpy.sqrt(r))),
        'p6': int(numpy.floor(Ng/numpy.sqrt(3.*r))),
        'pm': int(numpy.floor(Ng/numpy.sqrt(2.*r))),
        'pmm': int(numpy.floor(Ng/2./numpy.sqrt(r))),
        'p4m': int(numpy.floor(Ng/2./numpy.sqrt(r))),
        'p6m': int(numpy.floor(Ng/numpy.sqrt(6.*1))),
        'p3m1': int(numpy.floor(Ng/numpy.sqrt(3.*r))),
        'p31m': int(numpy.floor(Ng/numpy.sqrt(3.*r))),
        'p4g': int(numpy.floor(Ng/2./numpy.sqrt(r))),
        'cmm': int(numpy.floor(Ng/numpy.sqrt(2.*1))),
        'pmg': int(numpy.floor(Ng/2./numpy.sqrt(r))),
        'pg': int(numpy.floor(Ng/numpy.sqrt(2.*r))),
        'cm': int(numpy.floor(Ng/numpy.sqrt(2.*r))),
        'pgg': int(numpy.floor(Ng/2./numpy.sqrt(r)))
    }

    return N_points[group]

def _reduce_to_integers(fraction_list):
    """
    Multiplies a collection of fractions by constant values until they
    have been converted completely to integers.

    Parameters
    ----------
    fraction_list : list(fractions.Fraction)
        Fractions to reduce to integers.

    Returns
    -------
    list(int)
        Integers created by multiplying all provided fractions by a constant.
    """

    while True:
        largest_denominator = max(fraction_list, key=lambda fraction: fraction.denominator).denominator
        if largest_denominator == 1:
            return [int(fraction) for fraction in fraction_list]
        fraction_list = [fraction * largest_denominator for fraction in fraction_list]

# Holds information about a wallpaper group
class _WallpaperGroup(collections.namedtuple("WallpaperGroup", \
    ["number", "name", "symbol", "half", "dot", "ratio", "corners", "edges", "vectors", \
    "copies", "stoichiometry"])):
    """
    Represents a wallpaper group and contains information about the
    requirements on the polygon tiling space and the parallelogram
    necessary to recreate the tiling using translations alone.

    Parameters
    ----------
    number : int
        A unique number assigned to the wallpaper group.
    name : str
        The identifier, in Hermann-Mauguin notation, for the wallpaper
        group.
    symbol : str
        The identifier, in orbifold notation, for the wallpaper group.
    half : bool
        If the tiling polygon is a triangle rather than a parallelogram.
        If this is the case, the triangle is represented as the half of
        a parallelogram closest to the origin of its vectors.
    dot : float or None
        If not None, specifies that the dot product of the unit vectors
        of the parallelogram is restricted to a specific value.
    ratio : float or None
        If not None, specifies that the ratio of the magnitudes of the
        vectors of the parallelogram is restricted to a specific value.
    corners : list(set(int))
        Information regarding corner symmetry requirements.
    edges : list(_EdgeSymmetry)
        Information regarding edge symmetry requirements.
    vectors : list(tuple(float))
        Information regarding translating parallelogram (primitive cell) vector lengths
        relative to tiling polygon parallelogram (fundamental domain) vector lengths.
    copies : list(list(_PeriodicOperation))
        Information regarding copies of the tiling polygon (fundamental domain) that should
        be made during the creation of the translating parallelogram (primitive cell).
    stoichiometry : list(fractions.Fraction)
        Information regarding corner stoichiometry.
    """

    def __repr__(self):
        return "<paccs.minimization.WallpaperGroup at 0x{:x} as {}/{}/{} tiling {} {}{}>".format( \
            id(self), self.number, self.name, self.symbol, len(self.copies), \
            "triangle" if self.half else "parallelogram", "" if len(self.copies) == 1 else "s")

# Edge symmetry specification codes
class _EdgeSymmetry(_tuple): pass
class _FWD(_EdgeSymmetry): pass # Edges match in forward directions
class _REV(_EdgeSymmetry): pass # Edges match in reverse directions
class _INV(_EdgeSymmetry): pass # Edge has inversion symmetry about center

# Periodic image operation codes
class _PeriodicOperation(_tuple): pass
class _REF(_PeriodicOperation): pass # Inversion along (x, y) vectors in target coordinates
class _ROT(_PeriodicOperation): pass # Rotation (n) twelfth-turns counterclockwise about target origin
class _TRN(_PeriodicOperation): pass # Translation along (x, y) vectors of target

# The seventeen wallpaper groups
_wallpaper_groups = [
    _WallpaperGroup(1,   "p1",   "o",    False,  None,   None,   [{0, 1, 2, 3}],     [_FWD(0, 1), _FWD(2, 3)],       [(1.0, 0.0), (0.0, 1.0)],
        [[]], [_F(1, 4), _F(1, 4), _F(1, 4), _F(1, 4)]),
    _WallpaperGroup(2,   "p2",   "2222", False,  None,   None,   [{0, 1}, {2, 3}],   [_INV(0), _INV(1), _FWD(2, 3)], [(1.0, 0.0), (0.0, 2.0)],
        [[], [_REF(1, 1), _TRN(1.0, 1.0)]], [_F(1, 4), _F(1, 4), _F(1, 4), _F(1, 4)]),
    _WallpaperGroup(3,   "pm",   "**",   False,  0.0,    None,   [{0, 2}, {1, 3}],   [_FWD(0, 1)],                   [(2.0, 0.0), (0.0, 1.0)],
        [[], [_REF(1, 0), _TRN(1.0, 0.0)]], [_F(1, 4), _F(1, 4), _F(1, 4), _F(1, 4)]),
    _WallpaperGroup(4,   "pg",   "xx",   False,  0.0,    None,   [{0, 1, 2, 3}],     [_FWD(0, 1), _REV(2, 3)],       [(2.0, 0.0), (0.0, 1.0)],
        [[], [_REF(0, 1), _TRN(0.5, 1.0)]], [_F(1, 4), _F(1, 4), _F(1, 4), _F(1, 4)]),
    _WallpaperGroup(5,   "pmm",  "*2222",False,  0.0,    None,   [],                 [],                             [(2.0, 0.0), (0.0, 2.0)],
        [[], [_REF(1, 0), _TRN(1.0, 0.0)], [_REF(0, 1), _TRN(0.0, 1.0)], [_REF(1, 1), _TRN(1.0, 1.0)]], [_F(1, 4), _F(1, 4), _F(1, 4), _F(1, 4)]),
    _WallpaperGroup(6,   "pmg",  "22*",  False,  0.0,    None,   [{0, 2}, {1, 3}],   [_INV(2), _INV(3)],             [(2.0, 0.0), (0.0, 2.0)],
        [[], [_REF(1, 0), _TRN(1.0, 0.5)], [_REF(1, 1), _TRN(1.0, 0.5)], [_REF(0, 1), _TRN(0.0, 1.0)]], [_F(1, 4), _F(1, 4), _F(1, 4), _F(1, 4)]),
    _WallpaperGroup(7,   "pgg",  "22x",  False,  0.0,    None,   [{0, 3}, {1, 2}],   [_REV(0, 1), _REV(2, 3)],       [(2.0, 0.0), (0.0, 2.0)],
        [[], [_REF(1, 0), _TRN(0.5, 0.5)], [_REF(0, 1), _TRN(0.5, 0.5)], [_REF(1, 1), _TRN(1.0, 1.0)]], [_F(1, 4), _F(1, 4), _F(1, 4), _F(1, 4)]),
    _WallpaperGroup(8,   "cm",   "*x",   False,  0.0,    None,   [{0, 3}, {1, 2}],   [_REV(2, 3)],                   [(2.0, 0.0), (1.0, 1.0)],
        [[], [_REF(0, 1), _TRN(0.0, 1.0)]], [_F(1, 4), _F(1, 4), _F(1, 4), _F(1, 4)]),
    _WallpaperGroup(9,   "cmm",  "2*22", True,   0.0,    None,   [{1, 2}],           [_INV(4)],                      [(2.0, 0.0), (1.0, 1.0)],
        [[], [_REF(1, 1), _TRN(0.0, 1.0)], [_REF(1, 0), _TRN(1.0, 0.0)], [_REF(0, 1), _TRN(0.0, 1.0)]], [_F(1, 4), _F(1, 8), _F(1, 8)]),
    _WallpaperGroup(10,  "p4",   "442",  False,  0.0,    1.0,    [{1, 2}],           [_FWD(0, 2), _FWD(1, 3)],       [(2.0, 0.0), (0.0, 2.0)],
        [[], [_ROT(3.0), _TRN(1.0, 0.0)], [_ROT(6.0), _TRN(1.0, 1.0)], [_ROT(9.0), _TRN(0.0, 1.0)]], [_F(1, 4), _F(1, 4), _F(1, 4), _F(1, 4)]),
    _WallpaperGroup(11,  "p4m",  "*442", True,   0.0,    1.0,    [],                 [],                             [(2.0, 0.0), (0.0, 2.0)],
        [[], [_REF(1, 0), _TRN(1.0, 0.0)], [_REF(0, 1), _TRN(0.0, 1.0)], [_REF(1, 1), _TRN(1.0, 1.0)],
        [_ROT(3.0), _TRN(0.5, 0.5)], [_REF(1, 0), _ROT(3.0), _TRN(0.5, 0.5)], [_ROT(9.0), _TRN(0.5, 0.5)], [_REF(1, 0), _ROT(9.0), _TRN(0.5, 0.5)]],
        [_F(1, 4), _F(1, 8), _F(1, 8)]),
    _WallpaperGroup(12,  "p4g",  "4*2",  True,   0.0,    1.0,    [{1, 2}],           [_FWD(0, 2)],                   [(2.0, 0.0), (0.0, 2.0)],
        [[], [_ROT(3.0), _TRN(1.0, 0.0)], [_ROT(6.0), _TRN(1.0, 1.0)], [_ROT(9.0), _TRN(0.0, 1.0)],
        [_REF(1, 0), _TRN(0.5, 0.5)], [_REF(1, 0), _ROT(3.0), _TRN(0.5, 0.5)], [_REF(1, 0), _ROT(6.0), _TRN(0.5, 0.5)], [_REF(1, 0), _ROT(9.0), _TRN(0.5, 0.5)]],
        [_F(1, 4), _F(1, 8), _F(1, 8)]),
    _WallpaperGroup(13,  "p3",   "333",  False,  0.5,    1.0,    [{0, 3}],   [_REV(0, 3), _REV(1, 2)],       [(3 ** 0.5, 0.0), (-(3 ** 0.5), 3 ** 0.5)],
        [[_ROT(1.0)], [_ROT(5.0), _TRN(1.0, 0.0)], [_ROT(9.0), _TRN(0.0, 1.0)]], [_F(1, 6), _F(1, 3), _F(1, 3), _F(1, 6)]),
    _WallpaperGroup(14,  "p3m1", "*333", True,   0.5,    1.0,    [],                 [],                             [(3 ** 0.5, 0.0), (-(3 ** 0.5), 3 ** 0.5)],
        [[_ROT(1.0)], [_ROT(5.0)], [_ROT(9.0)], [_REF(1, 0), _ROT(3.0)], [_REF(1, 0), _ROT(7.0)], [_REF(1, 0), _ROT(11.0)]], [_F(1, 6), _F(1, 6), _F(1, 6)]),
    _WallpaperGroup(15,  "p31m", "3*3",  True,   -0.5,   1.0,    [{1, 2}],           [_FWD(0, 2)],                   [(3 ** 0.5, 0.0), (3 ** 0.5, 3 ** 0.5)],
        [[_ROT(3.0), _TRN(1 / 3, 1 / 3)], [_ROT(7.0), _TRN(1 / 3, 1 / 3)], [_ROT(11.0), _TRN(1 / 3, 1 / 3)],
        [_REF(1, 0), _ROT(3.0), _TRN(2 / 3, 2 / 3)], [_REF(1, 0), _ROT(7.0), _TRN(2 / 3, 2 / 3)], [_REF(1, 0), _ROT(11.0), _TRN(2 / 3, 2 / 3)]],
        [_F(1, 3), _F(1, 12), _F(1, 12)]),
    _WallpaperGroup(16,  "p6",   "632",  True,   -0.5,   1.0,    [{1, 2}],           [_FWD(0, 2), _INV(4)],          [(3 ** 0.5, 0.0), (3 ** 0.5, 3 ** 0.5)],
        [[_ROT(3.0), _TRN(1 / 3, 1 / 3)], [_ROT(7.0), _TRN(1 / 3, 1 / 3)], [_ROT(11.0), _TRN(1 / 3, 1 / 3)],
        [_ROT(1.0), _TRN(2 / 3, 2 / 3)], [_ROT(5.0), _TRN(2 / 3, 2 / 3)], [_ROT(9.0), _TRN(2 / 3, 2 / 3)]], [_F(1, 3), _F(1, 12), _F(1, 12)]),
    _WallpaperGroup(17,  "p6m",  "*632", True,   0.5,    2.0,    [],                 [],                             [(3 ** 0.5, 0.0), (0.0, 2 * (3 ** 0.5))],
        [[_ROT(3.0), _TRN(1 / 3, 1 / 3)], [_ROT(7.0), _TRN(1 / 3, 1 / 3)], [_ROT(11.0), _TRN(1 / 3, 1 / 3)],
        [_REF(1, 0), _ROT(1.0), _TRN(1 / 3, 1 / 3)], [_REF(1, 0), _ROT(5.0), _TRN(1 / 3, 1 / 3)], [_REF(1, 0), _ROT(9.0), _TRN(1 / 3, 1 / 3)],
        [_ROT(1.0), _TRN(2 / 3, 2 / 3)], [_ROT(5.0), _TRN(2 / 3, 2 / 3)], [_ROT(9.0), _TRN(2 / 3, 2 / 3)],
        [_REF(1, 0), _ROT(3.0), _TRN(2 / 3, 2 / 3)], [_REF(1, 0), _ROT(7.0), _TRN(2 / 3, 2 / 3)], [_REF(1, 0), _ROT(11.0), _TRN(2 / 3, 2 / 3)]],
        [_F(1, 6), _F(1, 12), _F(1, 4)])
]

def _wallpaper_stoichiometry(group):
    """
    Creates a list of stoichiometric factors for different types of positions
    within the tile of a wallpaper group: corner positions, off-center edge
    positions, center edge positions, and face positions.

    Parameters
    ----------
    group : WallpaperGroup
        The wallpaper group for which to generate stoichiometric factors.

    Returns
    -------
    list(int)
        A list of integer stoichiometric factors for the different types
        of positions within the tile of the group.
    """

    # Handle corner stoichiometry
    corner_stoichiometry = list(group.stoichiometry)
    for corner_index in range(len(corner_stoichiometry)):
        try:
            corner_specification = next((corner_specification for corner_specification in group.corners \
                if corner_index in corner_specification))
            # Assigning to this corner assigns to the other corners in the specification
            corner_stoichiometry[corner_index] *= len(corner_specification)
        except StopIteration:
            # This corner is not part of a symmetry; don't modify its factor
            pass

    # Handle edge stoichiometry for offset positions
    edge_offset_stoichiometry = [_F(1, 2) for index in range(3 if group.half else 4)]
    edge_center_stoichiometry = [_F(1, 2) for index in range(3 if group.half else 4)]
    for edge_index in range(len(edge_offset_stoichiometry)):

        # Deal with standard type (_FWD and _REV) cases
        try:
            # For triangles, edges are assigned (0, 2, 4) instead of (0, 1, 2, 3)
            edge_specification = next((edge_specification for edge_specification in group.edges \
                if not isinstance(edge_specification, _INV) and edge_index * (2 if group.half else 1) in edge_specification))
            # Assigning to this edge assigns to the other edge in the specification
            edge_offset_stoichiometry[edge_index] *= len(edge_specification)
            edge_center_stoichiometry[edge_index] *= len(edge_specification)
        except StopIteration:
            # This edge is not part of a symmetry; don't modify its factor
            pass

        # Deal with special (_INV) cases
        try:
            # For triangles, edges are assigned (0, 2, 4) instead of (0, 1, 2, 3)
            edge_specification = next((edge_specification for edge_specification in group.edges \
                if isinstance(edge_specification, _INV) and edge_index * (2 if group.half else 1) in edge_specification))
            # Assigning to this edge assigns to the other half of the edge ONLY for off-center nodes
            edge_offset_stoichiometry[edge_index] *= 2 * len(edge_specification)
        except StopIteration:
            # This edge is not part of a symmetry; don't modify its factor
            pass

    # Create integers from the fractions
    return _reduce_to_integers(corner_stoichiometry + edge_offset_stoichiometry + edge_center_stoichiometry + [_F(1, 1)])

# Pregenerate stoichiometric factors on module load
_wallpaper_stoichiometries = [_wallpaper_stoichiometry(group) for group in _wallpaper_groups]

def _retrieve_wallpaper_group(*, number=None, name=None, symbol=None):
    # This is a 3to2/Cython/Sphinx compatibility hack
    # Docstring is below in wrapper function

    # Check to make sure one specifier was given
    not_none = sum(1 for item in (number, name, symbol) if item is not None)
    if not_none != 1:
        raise ValueError("exactly one group specifier must be present")

    # Python 2 interface compatibility
    if name is not None and isinstance(name, bytes):
        name = name.decode()

    # Look up the wallpaper group from the table
    index, target = next(((index, target) for index, target in enumerate((number, name, symbol)) if target is not None))
    return next((group for group in _wallpaper_groups if getattr(group, ["number", "name", "symbol"][index]) == target))

def WallpaperGroup(*args, **kwargs):
    """
    Creates a wallpaper group object from either a number, Hermann-Mauguin
    notation name, or orbifold notation symbol.  Exactly one specifier should
    be selected.

    Parameters
    ----------
    number : int
        The number (between 1 and 17) of the wallpaper group.
    name : str
        A Hermann-Mauguin notation symmetry specifier.
    symbol : str
        An orbifold notation symmetry specifier.
    """

    # This is a 3to2/Cython/Sphinx compatibility hack
    # Implementation is above
    return _retrieve_wallpaper_group(*args, **kwargs)

def tile_wallpaper(cell, group, tolerance=1e-6, overlap_check=True):
    r"""
    From a fundamental domain and a wallpaper group, generates a minimal cell which tiles
    the plane via translation alone and contains all of the isometries
    in the group (aka the primitive cell).  This is used by :py:func:`generate_wallpaper`
    which generally provides a more user friendly interface.  This function raises a
    warning if unlike particles overlap as a result of these symmetry operations - if they are
    the same type, this is acceptable; however, if they are not, you probably
    have an error, so a warning is raised.

    Parameters
    ----------
    cell : paccs.crystal.Cell
        The two-dimensional cell to process (fundamental domain).
    group : WallpaperGroup
        A wallpaper group object (needs to be consistent with cell, or else will raise an error).
    tolerance : float
        Tolerance for various checks performed during the procedure.
    overlap_check : bool
        If True, will check if lattice sites (to within tolerance) are simultaneously
        occupied by different species. This can be very slow (doubles runtime), so avoid
        using this in production-level operation if possible.

    Raises
    ------
    ValueError
        Invalid cell or group specifications.

    Returns
    -------
    paccs.crystal.Cell
        The primitive cell.  Except in the case of the simplest group
        :math:`\mathrm{p1}`, the new vectors will be longer and the total number
        of atoms will have increased relative to the fundamental domain.
    """

    # Check dimensions
    if cell.dimensions != 2:
        raise ValueError("cell must be 2-dimensional to tile with wallpaper groups")

    # Check dot product
    if group.dot is not None:
        dot = numpy.dot(cell.vectors[0] / numpy.linalg.norm(cell.vectors[0]), \
            cell.vectors[1] / numpy.linalg.norm(cell.vectors[1]))
        if not numpy.isclose(dot, group.dot, rtol=tolerance, atol=tolerance):
            warnings.warn("angle between cell vectors should be {}deg; result may not possess desired symmetries" \
                .format(numpy.cos(dot)), RuntimeWarning)

    # Check magnitude ratio
    if group.ratio is not None:
        ratio = numpy.linalg.norm(cell.vectors[0]) / numpy.linalg.norm(cell.vectors[1])
        if not numpy.isclose(ratio, group.ratio, rtol=tolerance, atol=tolerance):
            warnings.warn("ratio of magnitudes of cell vectors should be {}; result may not possess desired symmetries" \
                .format(ratio), RuntimeWarning)

    # Make new vectors
    new_vectors = numpy.dot(cell.vectors.T, numpy.array(group.vectors).T).T

    # Make copies of atoms
    new_atom_lists = [numpy.zeros((cell.atom_count(type_index) * len(group.copies), cell.dimensions)) \
        for type_index in range(cell.atom_types)]
    for copy_index, copy in enumerate(group.copies):
        for type_index in range(cell.atom_types):
            # Pull out coordinates to modify
            coordinates = cell.atoms(type_index)

            # Go through and perform the transformations
            for transformation in copy:
                if isinstance(transformation, _REF):
                    # Inversion of coordinates about origin
                    coordinates *= [-1 if value else 1 for value in transformation]

                elif isinstance(transformation, _ROT):
                    # Rotation of coordinates about origin
                    angle = transformation[0] * numpy.pi / 6
                    cosine, sine = numpy.cos(angle), numpy.sin(angle)
                    coordinates = numpy.dot(numpy.array([[cosine, -sine], [sine, cosine]]), coordinates.T).T

                elif isinstance(transformation, _TRN):
                    # Translation of coordinates relative to origin
                    coordinates += numpy.dot(new_vectors.T, numpy.array(transformation).T).T

                else:
                    raise RuntimeError("internal failure: unsupported transformation")

            # Reassign modified coordinates
            atom_count = cell.atom_count(type_index)
            new_atom_lists[type_index][copy_index * atom_count:(copy_index + 1) * atom_count] = coordinates

    # Make new cell; check for unexpected overlaps between atoms of different types
    # Do not shift here or automatic duplicate detection in generation routine will break <-- this is deprecated, shiftinf still done later
    cell = crystal.CellTools.reduce(crystal.Cell(new_vectors, new_atom_lists, cell.names), tolerance=tolerance, shift=False)
    if (overlap_check):
        test_cell = crystal.CellTools.reassign(cell, {type_index: 0 for type_index in range(cell.atom_types)})
        if test_cell.contact(0, 0) < tolerance:
            raise RuntimeError("particles are very close or overlapping; cell may not have met symmetry specifications")

    return cell

def _1_log_Ni(Ni):
    return 1.0 + numpy.log(Ni)

def _solve_integer_problem(stoichiometry, place_min, place_max, nodes):
    """
    Formulates a stoichiometry problem as a constraint solving problem (CSP) and solves for all solutions.
    This is framed in a way such that all variables take on integer values.

    Parameters
    ----------
    stoichiometry : tuple(int)
        The desired final stoichiometries of the different atom types.
    place_min : int or None
        The total minimum number of (independent) nodes to place atoms at.
    place_max : int or None
        The total maximum number of (independent) nodes to place atoms at.
    nodes : list(tuple(int))
        A list of different node collections.  Each collection of nodes
        defined by a 2-tuple of integers should have a number of nodes
        and a stoichiometric contribution factor, respectively.

    Returns
    -------
    list(tuple(dict(tuple(int), int), int))
        A list of all solutions to the integer CSP.  Each solution's
        dictionary key/value pairs specify the number of atoms to place
        for given atom types and node collections, respectively.  The
        dictionary is accompanied by a value indicating the number of
        ways (based on combinatorics) to perform the arrangement indicated
        by the solution.
    """

    # Define the problem variables
    integer_problem = constraint.Problem(constraint.BacktrackingSolver())
    for type_index in range(len(stoichiometry)):
        for collection_index, collection in enumerate(nodes):
            # Give each variable its hard occupancy limit as an upper bound
            integer_problem.addVariable((type_index, collection_index), \
                list(range(collection[0] + 1)))

    # Enforce occupancy limits
    for collection_index, collection in enumerate(nodes):
        integer_problem.addConstraint(constraint.MaxSumConstraint(collection[0]), \
            [(type_index, collection_index) for type_index in range(len(stoichiometry))])

    # At least one atom of each type desired should be present
    for type_index, stoichiometric_coefficient in enumerate(stoichiometry):
        if stoichiometric_coefficient:
            integer_problem.addConstraint(constraint.MinSumConstraint(1), \
                [(type_index, collection_index) for collection_index in range(len(nodes))])

    # The total number of atoms placed should be in the desired range
    all_variables = list(itertools.product(range(len(stoichiometry)), range(len(nodes))))
    if place_min is not None:
        integer_problem.addConstraint(constraint.MinSumConstraint(place_min), all_variables)
    if place_max is not None:
        integer_problem.addConstraint(constraint.MaxSumConstraint(place_max), all_variables)

    # Linear equations for stoichiometry must be satisfied
    contribution = lambda type_index, variables: \
        sum(variables[(type_index * len(nodes)) + collection_index] * collection[1] \
        for collection_index, collection in enumerate(nodes))
    total_contribution = lambda variables: sum(contribution(type_index, variables) \
        for type_index in range(len(stoichiometry)))
    total_stoichiometry = sum(stoichiometry)
    for type_index, stoichiometric_coefficient in enumerate(stoichiometry):
        # Be sure to specify the variables explicitly to make sure the order is correct
        integer_problem.addConstraint(lambda *variables, type_index=type_index, \
            stoichiometric_coefficient=stoichiometric_coefficient: \
            contribution(type_index, variables) * total_stoichiometry == \
            total_contribution(variables) * stoichiometric_coefficient, all_variables)

    # The problem is now fully defined; solve it and do calculations
    problem_solutions = integer_problem.getSolutions()
    def combinatoric_count(problem_solution):
        total = 1
        # Find the product of the totals for each node collection
        for collection_index, collection in enumerate(nodes):
            # Prepare for calculation
            total_positions = collection[0]
            denominator = 1
            blank_positions = total_positions

            # Perform calculation for each atom type
            for type_index in range(len(stoichiometry)):
                blank_positions -= problem_solution[(type_index, collection_index)]
                denominator *= math.factorial(problem_solution[(type_index, collection_index)])
            total *= math.factorial(total_positions) // (denominator * math.factorial(blank_positions))
        return total
    return [(solution, combinatoric_count(solution)) for solution in problem_solutions]

def generate_wallpaper(stoichiometry, log_level=0, log_file=sys.stderr, place_min=1, \
    place_max=None, grid_count=5, angles=(0.5235987755982988, 0.7853981633974483, 1.0471975511965976, \
    1.5707963267948966), length_ratios=(1.0, 1.4142135623730951, 1.7320508075688772, 2.0), sample_count=None, \
    random_seed=None, weighting_function=_1_log_Ni, weighting_exponent=0.0, \
    merge_sets=True, sample_corners=True, sample_edge_offsets=True, sample_edge_centers=True, \
    sample_faces=True, sample_groups=None, tolerance=1e-6, count_configurations=False, \
    chosen_solution_idx=None, debug=False, congruent=False, minimum_configurations=0):
    r"""
    Yields primitive cell(s) containing one or more types of atoms
    based on given stoichiometry and grid parameters.  Symmetry groups
    are used to generate all possible tilings of two-dimensional space
    within the constraints of the specified grid.  Various parameters are
    available to customize the characteristics of the tilings created.
    Each structure returned is a unique solution to the constraint satisfaction
    problem, though this does not formally guarantee that all structures are
    truly unique.

    Parameters
    ----------
    stoichiometry : tuple(int)
        The stoichiometric ratio of the different types of atoms in each
        generated cell.
    log_level : int
        Verbosity of diagnostic messages to display.  By default, nothing is displayed.
        Increasing this value leads to more verbose output; setting it to -1 displays
        all messages.
    log_file : file
        Where to send diagnostic messages.  By default, they are sent to the standard
        error stream.
    place_min : int
        The minimum number of nodes to place atoms at on a tile.  By default, this
        value is set to 1.  Typically, more atoms than specified will be
        needed to achieve the desired stoichiometry. Formally, a minimum of 1 node
        must be occupied to meet stoichiometric requirements which is built in
        automatically anyway.
    place_max : int
        The maximum number of independent nodes to place atoms at on a tile.  By default, no
        limit is present.  If the desired stoichiometry cannot be achieved
        within the specified bounds on the number of atoms placed, then no
        results will be yielded.
    grid_count : int
        The target number of grid nodes along the edge of a tile, including endpoints.
        Depending on the constraints or
        lack thereof imposed by any given symmetry group, it may be the
        case that more than this number of nodes are created along a
        given side of the tile.  However, at least this many nodes will
        be present along each side.  Performance may suffer if this value
        is extremely large. Also see **congruent** - if this is True, then grid_count
        refers to the "congruent" :math:`p1` cell not the cell of each wallpaper group used.
    angles : tuple(float)
        For wallpaper groups :math:`\mathrm{p1}` and :math:`\mathrm{p2}`,
        the angle :math:`\theta` between the two vectors of the tile is
        variable.  This parameter permits specification of the values of
        :math:`\theta` used.  Radians are expected.
    length_ratios : tuple(float)
        For wallpaper groups :math:`\mathrm{p1}`, :math:`\mathrm{p2}`,
        :math:`\mathrm{pm}`, :math:`\mathrm{pg}`, :math:`\mathrm{pmm}`,
        :math:`\mathrm{pmg}`, :math:`\mathrm{pgg}`, :math:`\mathrm{cm}`,
        and :math:`\mathrm{cmm}`, the ratio of the lengths :math:`a`
        and :math:`b` of the two vectors of the tile is variable.
        This parameter permits specification of the values of :math:`a/b` used.
    sample_count : int
        By default, this will attempt to provide all possible configurations
        meeting the specifications.  However, specifying a value for this parameter
        will cause no more than that number of configurations to be generated.
        This is not a guarantee that this many will be generated, only that no
        more than this will be. They will be chosen from the set of all
        allowable configurations.
    random_seed : int
        The seed for the random number generator (which uses the Mersenne Twister algorithm).
        If not specified, a random seed will be chosen.  Specify this value if it
        is desired to generate exactly reproducible results.  It should be non-negative
        and less than :math:`2^{32}`.
    weighting_function : callable
        A function that can be used to bias the configurations being sampled, in
        conjuction with **weighting_exponent**. Upon the first call to the generator,
        the (integer) constraint solving problem corresponding
        to the grid and stoichiometry specified is solved.  This produces some number of
        solutions each having a number of possible final configurations :math:`N_1, N_2, ..., N_i, ...`.
        The probability of selecting from each configuration is :math:`p(i)=f(N_i)`.  By default,
        :math:`f(N_i)=1+\log N_i`, but this parameter allows for custom weighting functions.
        The callable should accept an :py:class:`int` and return a :py:class:`float`.  When using
        a :py:mod:`paccs.automation.TaskManager`, this function should be defined
        directly using a *def* at the module level (such that it is importable by worker processes).
    weighting_exponent : float
        The weighting function is raised to this power and used (after normalization)
        to directly generate probabilities of configurations. This can be useful, for
        instance, to bias towards selecting configuration sets containing fewer
        configurations (by choosing a negative value).  By default, this value is
        set to 0 so that all sets of (groups, solutions) will be chosen with equal probability (no bias).
    merge_sets : bool
        If this option is set to True, then sets of stoichiometrically equivalent configurations
        will be merged into the same node category. For example, if this option
        is set, atoms placed on edges with the same stoichiometric contribution as atoms
        placed within tiles will be treated identically with respect to atom assignment.
        Sets of configurations with different assignments to these nodes will not
        be treated separately.
    sample_corners : bool
        Whether or not to yield configurations with atoms on corners of tiles (fundamental domains).
    sample_edge_offsets : bool
        Whether or not to yield configurations with atoms on sides of edges of tiles (fundamental domains).
    sample_edge_centers : bool
        Whether or not to yield configurations with atoms on centers of edges of
        tiles (fundamental domains).  These positions will only exist when an odd number of nodes are
        present along a given edge.
    sample_faces : bool
        Whether or not to yield configurations with atoms in the centers (faces) of tiles (fundamental domains).
    sample_groups : list(WallpaperGroup)
        The wallpaper groups from which to generate configurations.  If not specified,
        all 17 wallpaper groups will be used.
    tolerance : float
        The tolerance used for various checks and comparisons during the process.
    count_configurations : bool
        Instead of yielding cells, generate lookup tables, yield the maximum number
        of configurations that could theoretically be yielded, and stop.  This can be
        useful to get a bound on **sample_count**.
    chosen_solution_idx : int
        The user can force a configuration to be selected from a given solution to the
        stoichiometric problem if desired.  This index should be between
        0 <= chosen_solution_idx < :math:`N_{\rm sol}`, but is not sorted in any particular way.
        By default, this is None, indicating to choose a solution index at random, if configurations are desired.
        This is often just for debugging purposes, or verbose investigation in certain cases, such as
        explicitly enumerating all configurations a single time. **This is intended to be used with only one allowable group at a time.**
        Note that this also implies only a single angle/length ratio should be given a time since internally
        the same wallpaper group with different values is treated as a different "effective group".
    debug : bool
        Flag that enables debugging operations, such as returning fundamental domain
        Cell objects instead of primitive cells.
    congruent : bool
        Flag indicating if it is desired that the grid spacing for each sampled group should be
        "congruent" with a p1 cell with that nodal density. If True, then grid_count is treated
        as being that of the p1 cell, **not** that of the groups requested (except p1), where Ng
        for each group is computed using :py:func:`paccs.wallpaper.Nx` instead.
    minimum_configurations : int
        Minimum number of configurations to enforce.  If not enough configurations exist, summed
        over all groups, angles, length rations, etc., then the grid_count will be increased
        incrementally until this many total configurations are found.  If **count_configurations**
        is True, this will not have any effect as it is assumed the user is counting configurations
        for a specific set of parameters.

    Returns
    -------
    generator(tuple(WallpaperGroup, paccs.crystal.Cell))
        Primitive cells containing arrangements of atoms conforming to the specified parameters.
        The group from which each cell was generated is provided with the cell itself.

    Notes
    -----
    The spacing of the grid will always be at least unity.  Consequently, the tile
    (fundamental domain) is always tesselated with parallelograms such that the
    minimum distance between nodes is at least unity.
    If a custom spacing is required, use :py:func:`paccs.crystal.CellTools.scale`
    after the generator returns the cell object.  It may be useful to generate a
    scale factor from :py:func:`paccs.crystal.Cell.scale_factor` in conjunction with
    known atomic radii.

    When all unique configurations have been returned, the generator will StopIteration.
    No repeated realizations to solutions of the CSP will be returned, although it is possible
    that different solutions/realizations may represent the same actual structure.
    """

    start_time, start_clock = time.time(), time.process_time()

    # Set up diagnostic output
    def log(level, message):
        if level > log_level and log_level != -1:
            return
        make_stars = message.startswith("@") and message != "@"
        if make_stars:
            log_file.write("*".join(" " for index in range(level + 1)))
        if message.endswith("@"):
            log_file.write("{}\n".format(message[int(make_stars):-1]))
        else:
            log_file.write(message[int(make_stars):])
            log_file.flush()
    log(1, "@Preparing for generation@")

    # Check angle and length ranges
    cos_angles = numpy.cos(numpy.array(angles))
    length_ratios = numpy.array(length_ratios)

    # Define cell vector function
    def generate_vectors(cos_angle, length_ratio):
        return numpy.array([[length_ratio, 0], [cos_angle, numpy.sqrt(1 - (cos_angle ** 2))]])

    # Are differing discretizations permitted for a given group?
    def check_grid_differing(group):
        return not group.half and not (\
            (0, 2) in group.edges or (0, 3) in group.edges or \
            (1, 2) in group.edges or (1, 3) in group.edges or \
            (2, 0) in group.edges or (2, 1) in group.edges or \
            (3, 0) in group.edges or (3, 1) in group.edges)

    # Solve discretization/symmetry problem for each group geometry
    master_grid_count = grid_count-1 # Decrement before loop increments once
    configuration_count = -1

    while (configuration_count < minimum_configurations):
        master_grid_count += 1
        log(1, "@Creating structure lookup table for grid_count={}@".format(master_grid_count))
        group_geometries = []
        for group in sample_groups if sample_groups is not None else _wallpaper_groups:
            for cos_angle in [group.dot] if group.dot is not None else cos_angles:
                for length_ratio in [group.ratio] if group.ratio is not None else length_ratios:
                    log(2, "@Working on {} with dot={}, ratio={}@".format(group, cos_angle, length_ratio))
                    grid_differing = check_grid_differing(group)

                    # Check if grid density should change
                    if (congruent):
                        # Take ratio > 1 so that grid number returned is the "minimum" a side with have
                        grid_count = Nx(master_grid_count, group.name, length_ratio if length_ratio >= 1 else 1.0/length_ratio)
                    else:
                        grid_count = master_grid_count
                    log(2, "@grid_count for this group and ratio is {}".format(grid_count))

                    # Retrieve stoichiometry list and indices into it
                    group_stoichiometry = _wallpaper_stoichiometries[group.number - 1]

                    # These offsets are so that different lattice site types receive
                    # unique indices (not necessarily contiguous) based on the
                    # enumeration scheme in Step 2 below.
                    stoichiometry_index_corners = 0
                    stoichiometry_index_offsets = 3 if group.half else 4
                    stoichiometry_index_centers = 6 if group.half else 8
                    stoichiometry_index_face = len(group_stoichiometry) - 1

                    # Step 1: solve the discretization problem, figuring out the number of nodes
                    if not grid_differing:
                        # The discretization must be the same along both axes
                        grid_count_x, grid_count_y = grid_count, grid_count
                        if length_ratio < 1:
                            # The x-axis is shorter than the y-axis
                            # Let the delta on the y-axis be longer than specified
                            grid_delta_x = 1.0
                            length_x = (grid_count_x - 1) * grid_delta_x
                            length_y = length_x / length_ratio
                            grid_delta_y = length_y / (grid_count_y - 1)
                        else:
                            # The y-axis is shorter than the x-axis
                            # Let the delta on the x-axis be longer than specified
                            grid_delta_y = 1.0
                            length_y = (grid_count_y - 1) * grid_delta_y
                            length_x = length_y * length_ratio
                            grid_delta_x = length_x / (grid_count_x - 1)
                    else:
                        # The discretization can be different in the two directions
                        # Begin by supposing that all criteria are met on the x-axis
                        grid_count_x, grid_delta_x = grid_count, 1.0
                        length_x = (grid_count_x - 1) * grid_delta_x
                        length_y = length_x / length_ratio
                        grid_count_y = int(numpy.floor(length_y / grid_delta_x)) + 1

                        # Check to make sure that the parity has been preserved
                        if grid_count_y % 2 != grid_count_x % 2:
                            # Decrease by 1 to make sure delta y is not less than delta
                            grid_count_y -= 1

                        # Test the assumption
                        if grid_count_y < grid_count_x:
                            # This assumption failed (insufficient number of points)
                            # Try to suppose that all criteria are met on the y-axis
                            grid_count_y, grid_delta_y = grid_count, 1.0
                            length_y = (grid_count_y - 1) * grid_delta_y
                            length_x = length_y * length_ratio
                            grid_count_x = int(numpy.floor(length_x / grid_delta_y)) + 1

                            # Check to make sure that the parity has been preserved
                            if grid_count_x % 2 != grid_count_y % 2:
                                # Decrease by 1 to make sure delta x is not less than delta
                                grid_count_x -= 1

                            # One or the other assumption should work: thus this one must
                            grid_delta_x = length_x / (grid_count_x - 1)
                        else:
                            # This assumption was successful
                            grid_delta_y = length_y / (grid_count_y - 1)

                    if (numpy.min([grid_delta_x, grid_delta_y]) < 1.0):
                        log(2, "@Unable to grid the fundamental domain: grid_delta_x,grid_delta_y={},{}@".format(grid_delta_x, grid_delta_y))
                        continue
                    if (numpy.min([grid_count_x, grid_count_y]) < 2):
                        log(2, "@Unable to grid the fundamental domain: grid_count_x,grid_count_y={},{}@".format(grid_count_x, grid_count_y))
                        continue
                        #raise Exception("failed to grid the fundamental domain properly")

                    # Step 2: build the grid and solve the symmetry problem
                    log(2, ", grid={}x{}, delta={}x{}@".format(grid_count_x, grid_count_y, grid_delta_x, grid_delta_y))
                    grid = numpy.swapaxes(numpy.rollaxis(numpy.array(numpy.meshgrid(numpy.linspace(0, 1, grid_count_x), \
                        numpy.linspace(0, 1, grid_count_y))), 0, start=3), 0, 1)
                    grid_fill, grid_copy = {}, {}

                    if sample_faces:
                        # Process internal nodes
                        assignment_count = 0
                        for index_x, index_y in itertools.product(range(1, grid_count_x - 1), range(1, grid_count_y - 1)):
                            if not group.half or index_x + index_y < grid_count_x - 1: # == grid_count_y - 1 if group.half
                                grid_fill[(index_x, index_y)] = stoichiometry_index_face
                                assignment_count += 1
                        log(3, "@Node assignments: internal={}".format(assignment_count))

                    if sample_corners:
                        # Process corners                             Corner index in group table
                        corners = [
                            (0, 0),                                 # 0
                            (grid_count_x - 1, 0),                  # 1
                            (0, grid_count_y - 1),                  # 2
                            (grid_count_x - 1, grid_count_y - 1)]   # 3

                        # Corners in symmetry specifications
                        assignment_count = 0
                        for corner_specification in group.corners:
                            retained_corner = min(corner_specification)
                            grid_fill[corners[retained_corner]] = stoichiometry_index_corners + retained_corner
                            for other_corner in corner_specification - {retained_corner}:
                                grid_copy[corners[other_corner]] = corners[retained_corner]
                            assignment_count += 1
                        log(3, ", corner_symm={}".format(assignment_count))

                        # Corners not in symmetry specifications
                        assignment_count = 0
                        for corner in set(range(len(corners) - (1 if group.half else 0) )) - \
                            functools.reduce(set.__or__, group.corners, set()):
                            grid_fill[corners[corner]] = stoichiometry_index_corners + corner
                            assignment_count += 1
                        log(3, ", corner_asymm={}".format(assignment_count))

                    if sample_edge_offsets or sample_edge_centers:
                        # Process edges                                                                       Edge index in group table
                        edges = [
                            [(index, 0) for index in range(1, grid_count_x - 1)],                           # 0
                            [(index, grid_count_y - 1) for index in range(1, grid_count_x - 1)],            # 1
                            [(0, index) for index in range(1, grid_count_y - 1)],                           # 2
                            [(grid_count_x - 1, index) for index in range(1, grid_count_y - 1)],            # 3
                            [(grid_count_x - 1 - index, index) for index in range(1, grid_count_x - 1)]]    # 4

                        # Find the indices of the centers of all edges (None if the center does not exist)
                        edge_centers = [(len(edge) - 1) // 2 if len(edge) != 0 and len(edge) % 2 == 1 else None for edge in edges]
                        # Determine based on the sampling criteria whether or not to include each edge node
                        edge_inclusions = [[(sample_edge_offsets and index != edge_centers[edge_index]) or \
                            (sample_edge_centers and index == edge_centers[edge_index]) for index in range(len(edge))] \
                            for edge_index, edge in enumerate(edges)]
                        # Edges in symmetry specifications
                        assignment_count = 0
                        for edge_specification in group.edges:
                            if isinstance(edge_specification, _FWD):
                                # Edges should match along forward directions
                                for index in range(len(edges[edge_specification[0]])):
                                    if edge_inclusions[edge_specification[0]][index]:
                                        stoichiometry_index = stoichiometry_index_centers \
                                            if index == edge_centers[edge_specification[0]] else stoichiometry_index_offsets
                                        grid_fill[edges[edge_specification[0]][index]] = \
                                            stoichiometry_index + (edge_specification[0] // (2 if group.half else 1))
                                        grid_copy[edges[edge_specification[1]][index]] = \
                                            edges[edge_specification[0]][index]
                                        assignment_count += 1

                            elif isinstance(edge_specification, _REV):
                                # Edges should match in reverse directions
                                for index in range(len(edges[edge_specification[0]])):
                                    if edge_inclusions[edge_specification[0]][index]:
                                        stoichiometry_index = stoichiometry_index_centers \
                                            if index == edge_centers[edge_specification[0]] else stoichiometry_index_offsets
                                        grid_fill[edges[edge_specification[0]][index]] = \
                                            stoichiometry_index + (edge_specification[0] // (2 if group.half else 1))
                                        grid_copy[edges[edge_specification[1]][len(edges[edge_specification[1]]) - 1 - index]] = \
                                            edges[edge_specification[0]][index]
                                        assignment_count += 1

                            elif isinstance(edge_specification, _INV):
                                for index in range(len(edges[edge_specification[0]]) // 2 + (0 if len(edges[edge_specification[0]]) % 2 == 0 else 1)):
                                    if edge_inclusions[edge_specification[0]][index]:
                                        stoichiometry_index = stoichiometry_index_centers \
                                            if index == edge_centers[edge_specification[0]] else stoichiometry_index_offsets
                                        grid_fill[edges[edge_specification[0]][index]] = \
                                            stoichiometry_index + (edge_specification[0] // (2 if group.half else 1))
                                        grid_copy[edges[edge_specification[0]][len(edges[edge_specification[0]]) - 1 - index]] = \
                                            edges[edge_specification[0]][index]
                                        assignment_count += 1

                            else:
                                raise RuntimeError("internal failure: unsupported symmetry")
                        log(3, ", edge_symm={}".format(assignment_count))

                        # Edges not in symmetry specifications
                        assignment_count = 0
                        for edge in set(range(0, 6, 2) if group.half else range(4)) - \
                            functools.reduce(set.__or__, [set(edge) for edge in group.edges], set()):
                            for index in range(len(edges[edge])):
                                if edge_inclusions[edge][index]:
                                    stoichiometry_index = stoichiometry_index_centers \
                                        if index == edge_centers[edge] else stoichiometry_index_offsets
                                    grid_fill[edges[edge][index]] = stoichiometry_index + (edge // (2 if group.half else 1))
                                    assignment_count += 1
                        log(3, ", edge_asymm={}".format(assignment_count))
                    log(3, "@")

                    # Step 3: solve the integer constraint solving problem
                    if merge_sets:
                        # Make groups for solving integer program based on stoichiometric factor alone
                        assignment_classes = {key: group_stoichiometry[value] for key, value in grid_fill.items()}
                        counter = collections.Counter(assignment_classes.values())
                        problem_specification = [(value, key) for key, value in counter.items()]
                        counter_keys = list(counter.keys())
                        solution_key = {key: counter_keys.index(value) for key, value in assignment_classes.items()}
                    else:
                        # Make groups for solving integer program based on stoichiometric factor and node type
                        assignment_classes = {key: (group_stoichiometry[value], value // (3 if group.half else 4)) \
                            for key, value in grid_fill.items()}
                        # collections.Counter is a dict: iteration order is arbitrary but guaranteed constant per unchanged instance
                        counter = collections.Counter(assignment_classes.values())
                        problem_specification = [(value, key[0]) for key, value in counter.items()]
                        counter_keys = list(counter.keys())
                        solution_key = {key: counter_keys.index(value) for key, value in assignment_classes.items()}
                    # Call the solver
                    solutions = _solve_integer_problem(stoichiometry, place_min, place_max, problem_specification)
                    log(3, "@Found {} solutions to stoichiometry problem@".format(len(solutions)))
                    # Display debug solver output if requested
                    if log_level == -1 or log_level >= 4:
                        log(4, "@{}@".format("|".join("{}=>{}".format(";".join("[{},{}->{}]".format(*key, value) \
                            for key, value in solution[0].items()), solution[1]) for solution in solutions)))

                    # Map the solution keys to integers for lattice enumeration
                    solution_key_map = {}
                    for pt in solution_key:
                        idx = solution_key[pt]
                        if idx in solution_key_map:
                            solution_key_map[idx].append(pt)
                        else:
                            solution_key_map[idx] = [pt]

                    # Save the results for this specific generation geometry
                    # group, vectors: defines the geometry of the tile
                    # grid: array of fractional coordinates of nodes in the grid
                    # grid_fill: assignable node indices -> types of all nodes (# of independent positions)
                    # grid_copy: virtual node indices -> mappings to assignable nodes (maps symmetry-equivalent sites)
                    # solution_key: assignable node indices -> key indices
                    # solution_key_map: key indices -> (indexed) list of points which have this key index, specific order is irrelevant
                    # solutions: array of solutions to the integer constraint solving problem
                    #     ( { (atom type, key index) -> number to place down }, number of ways to do it )
                    group_geometries.append((group, length_y * generate_vectors(cos_angle, length_ratio), \
                        grid, grid_fill, grid_copy, solution_key, solution_key_map, solutions))

        # Check to make sure that not too many configurations have been requested
        configuration_counts = [[solution[1] for solution in group_geometry[-1]] for group_geometry in group_geometries]
        configuration_count = sum(sum(count_list) for count_list in configuration_counts)
        log(1, "@Found a total of {} possible configurations@".format(configuration_count))

        # If only counting configurations, disregard any enforce command
        if count_configurations:
            yield configuration_count
            return

    if sample_count is None:
        sample_count = configuration_count
    if sample_count > configuration_count:
        warnings.warn("too many configurations were requested (not this many exist)", RuntimeWarning)
        sample_count = configuration_count
    log(1, "@Generating {} configurations@".format(sample_count))
    if (sample_count == 0):
        return

    # Generate configurations stochastically: prepare probabilities and PRNG
    weighted_counts = [[weighting_function(count) ** weighting_exponent for count in count_list] \
        for count_list in configuration_counts]
    weighted_count = sum(sum(count_list) for count_list in weighted_counts)
    probabilities = [[count / weighted_count for count in count_list] \
        for count_list in weighted_counts]
    probability_sums = [sum(probability_list) for probability_list in probabilities]
    normalized_probabilities = [[probabilities[list_index][index] / probability_sums[list_index] \
        for index in range(len(probabilities[list_index]))] \
        for list_index in range(len(probabilities))]
    randomness = numpy.random.RandomState(random_seed)

    # Generate flat lookup into nested array of probabilities
    probabilities_index = [(outer_index, inner_index) \
        for outer_index, count_list in enumerate(probabilities) \
        for inner_index in range(len(count_list))]

    # Generate psuedo-unique structures by choosing unique realizations of different solutions
    enumerated_lattices = {}
    not_fully_exhausted = {gr_index:[True]*len(normalized_probabilities[gr_index]) for gr_index in range(len(probability_sums))}
    group_choice_order = sorted(enumerate(probability_sums), key=lambda x:x[1])
    group_choice_idx_map = {group_choice[0]:group_choice_idx \
                        for group_choice_idx,group_choice in enumerate(group_choice_order)}
    solution_choice_order = {}
    solution_choice_idx_map = {}
    for gr_index in range(len(probability_sums)):
        solution_choice_order[gr_index] = sorted(enumerate(normalized_probabilities[gr_index]), key=lambda x:x[1])
        solution_choice_idx_map[gr_index] = {solution_choice[0]:solution_choice_idx \
                        for solution_choice_idx,solution_choice in \
                        enumerate(solution_choice_order[gr_index])}
    max_attempts = numpy.sum([len(normalized_probabilities[gr_index]) for gr_index in range(len(probability_sums))])#numpy.max([len(normalized_probabilities[gr_index]) for gr_index in range(len(probability_sums))])
    total_configurations_provided = 0.0
    for configuration_index in range(sample_count):
        log(2, "@Configuration {} of {}@".format(configuration_index + 1, sample_count))
        for attempt_index in range(max_attempts):
            # Allow loop to try again if it just exhausted a particular solution to a group.
            # Worst case scenario is that a group is chosen which has already found all its
            # unique solutions exactly, but since it needs to be called once more for the generator
            # to return a StopIteration command we need to allow the loop as many tries as the
            # max number of solutions exist for a group.
            log(3, "@Attempt {} of {}@".format(attempt_index + 1, max_attempts))

            # Choose a random group (if group has multiple length and/or angle ratios each instance is considered a "different group")
            random_group_index = randomness.choice(range(len(probability_sums)), p=probability_sums)
            group_loop_idx = 0
            while ((not numpy.any(not_fully_exhausted[random_group_index])) \
                and (group_loop_idx < len(probability_sums))):
                # If this group has no remaining unique solutions choose a new group
                group_loop_idx += 1
                next_guess = group_choice_idx_map[random_group_index] - 1 # Move toward lower probability
                if (next_guess < 0):
                    next_guess = len(probability_sums)-1
                random_group_index = group_choice_order[next_guess][0]
            if (group_loop_idx == len(probability_sums)):
                # Could not find a group with any remaining unused solutions
                log(3, "@Unable to find an unexhausted group@")
                configuration_index = sample_count
                break
                #raise StopIteration

            group, vectors, grid, grid_fill, grid_copy, solution_key, solution_key_map, solutions = group_geometries[random_group_index]
            log(3, "@Selecting a configuration from group {}@".format(group))

            # Choose a random solution from within that group
            if (chosen_solution_idx is None):
                random_solution_index = randomness.choice(range(len(normalized_probabilities[random_group_index])), \
                    p=normalized_probabilities[random_group_index])
                solution_loop_idx = 0
                while ((not not_fully_exhausted[random_group_index][random_solution_index]) \
                    and (solution_loop_idx < len(normalized_probabilities[random_group_index]))):
                    # If this solution has no remaining solutions choose a new solution
                    solution_loop_idx += 1
                    next_guess = solution_choice_idx_map[random_group_index][random_solution_index] - 1 # Move toward lower probability
                    if (next_guess < 0):
                        next_guess = len(normalized_probabilities[random_group_index])-1
                    random_solution_index = solution_choice_order[random_group_index][next_guess][0]
                if (solution_loop_idx == len(normalized_probabilities[random_group_index])):
                    # If a group was chosen, it was supposed to have some solutions left
                    raise Exception("failed to catch that all solutions were exhausted from a chosen group")
            else:
                if (chosen_solution_idx < 0 or chosen_solution_idx >= len(normalized_probabilities[random_group_index])): raise ValueError("invalid chosen_solution_idx for group {}".format(group))
                random_solution_index = chosen_solution_idx
            log(3, "@Selecting a configuration from stoichiometric solution index {}@".format(random_solution_index))
            solution = solutions[random_solution_index][0]

            # Get a unique configuration from that solution if possible
            if (random_group_index not in enumerated_lattices):
                enumerated_lattices[random_group_index] = {}
            if (random_solution_index not in enumerated_lattices[random_group_index]):
                chunks = dict([(site_type, len(solution_key_map[site_type])) for site_type in solution_key_map])
                recipe = {}
                for k in solution:
                    atom_type, site_type = k
                    if (site_type not in recipe):
                        recipe[site_type] = {}
                    if (atom_type not in recipe[site_type] and solution[k] > 0):
                        recipe[site_type][atom_type] = solution[k]
                ec = enumconfig.EnumeratedLattice(chunks, recipe)
                enumerated_lattices[random_group_index][random_solution_index] = ec.ran(seed=random_seed, clone=False)

            passed = True
            try:
                config = next(enumerated_lattices[random_group_index][random_solution_index])
            except StopIteration:
                passed = False
                if (chosen_solution_idx != None):
                    # Have exhausted all solutions in this (single) manually specified index, so terminate
                    # This loop is really just for debugging purposes - it halts after a group's solution is exhausted (assumes only using one group so no need to continue after this point)
                    log(1, "@Exhausted all unique configurations in chosen_solution_idx {} in group {}@".format(chosen_solution_idx, group))
                    raise StopIteration
                else:
                    # Can choose another solution index and/or group and try again (this one exhausted, but there might be others).
                    not_fully_exhausted[random_group_index][random_solution_index] = False

            if (passed):
                # Fill unique positions
                filled_grid = {}
                for site_type in config:
                    chunk_instruct = config[site_type]
                    if (len(chunk_instruct) != len(solution_key_map[site_type])):
                        raise Exception("mismatch between enumeration and CSP solution")
                    for i in range(len(chunk_instruct)):
                        if (chunk_instruct[i] != enumconfig.__EMPTY_SITE__):
                            filled_grid[solution_key_map[site_type][i]] = int(chunk_instruct[i])

                # Generate atom coordinate arrays, making copies for symmetry
                atom_coordinates = [[] for type_index in range(len(stoichiometry))]
                for key, value in filled_grid.items():
                    atom_coordinates[value].append(numpy.dot(vectors.T, grid[key].T).T)
                for key, value in grid_copy.items():
                    if value in filled_grid:
                        atom_coordinates[filled_grid[value]].append(numpy.dot(vectors.T, grid[key].T).T)

                # Build the configuration as a cell object
                new_cell = crystal.Cell(vectors, [numpy.array(atom_list) for atom_list in atom_coordinates])
                total_configurations_provided += 1.0

                if (debug):
                    # Return just the fundamental domain.
                    yield (group, new_cell)
                else:
                    # Shift here to provide a fully normalized form of the cell (puts origin at particle 0 of type 0).
                    # Return a primitive cell.
                    next_configuration = tile_wallpaper(new_cell, group, tolerance, overlap_check=False)
                    yield (group, crystal.CellTools.shift(next_configuration))
                break
        else:
            warnings.warn("failed to find unique configuration in allowable number of attempts", RuntimeWarning)
            break

    end_time, end_clock = time.time(), time.process_time()
    log(1, "@Total configurations provided by generator: {}@".format('%0.f'%total_configurations_provided))
    log(1, "@Generation routine wall time t={}s@".format(end_time - start_time))
    log(1, "@Generation routine processor time t={}s@".format(end_clock - start_clock))
