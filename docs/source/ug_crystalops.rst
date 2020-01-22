.. _ug_crystalops:

Periodic cell manipulation and visualization tools
==================================================

This guide walks through some common uses for features in the
:py:mod:`~paccs.crystal`, :py:mod:`~paccs.potential`, and
:py:mod:`~paccs.visualization` modules.

Working with cells
------------------

:py:class:`paccs.crystal.Cell` objects represent translationally
periodic cells having arbitrary vectors in any number of dimensions, containing
particles of one or more types.  Cell objects are easy to create::

   >>> cell = crystal.Cell(numpy.eye(3),
   ...     [numpy.zeros((1, 3)), 0.5 * numpy.ones((1, 3))])
   >>> cell
   <paccs.crystal.Cell in 3D with 1 A, 1 B>

The first argument should be a :math:`D\times D` square matrix of cell vectors,
and the second argument should be a collection of :math:`N_i\times D` matrices
for each particle type, where :math:`D` is the number of dimensions of the
space in which the cell exists, and :math:`N_i` is the number of particles of
the given type :math:`i`.  Here is another example::

   >>> cell2 = crystal.Cell(numpy.eye(2),
   ...     [numpy.zeros((1, 2)), 0.5 * numpy.eye(2)], ["X", "Y"])
   >>> cell2
   <paccs.crystal.Cell in 2D with 1 X, 2 Y>

Note the optional third argument that is accepted to give particle types custom
names.

.. note::
   paccs uses the convention of storing all vectors as row vectors, not
   column vectors.  Thus, if ``M`` is a matrix of cell vectors, ``M[i]``
   retrieves the row vector with index ``i`` and ``M[i, j]`` retrieves the
   component with index ``j`` from that row vector.  Be aware of this when
   manipulating matrices for input into paccs.

Basic properties
^^^^^^^^^^^^^^^^

Cell objects are immutable; it is not possible to modify them after they have
been created.  This allows them to be used as keys in dictionaries or elements
in sets.  However, it is possible to retrieve a variety of attributes about the
vectors::

   >>> cell.dimensions
   3
   >>> cell.vectors
   array([[ 1., 0., 0.],
          [ 0., 1., 0.],
          [ 0., 0., 1.]])
   >>> cell.vector(1)
   array([ 0., 1., 0.])

and about the particles (the API uses ``atom`` in place of ``particle`` for
brevity)::

   >>> cell2.atom_types
   2
   >>> cell2.atom_counts
   [1, 2]
   >>> cell2.atom_count(1)
   2
   >>> cell2.atom_lists
   [array([[ 0., 0.]]),
    array([[ 0.5,  0. ],
           [ 0. ,  0.5]])]
   >>> cell2.atoms(1)
   array([[ 0.5,  0. ],
          [ 0. ,  0.5]])
   >>> cell2.atom(1, 1)
   array([ 0. ,  0.5])
   >>> cell2.names
   ['X', 'Y']

Note that particle type names are optional, and if unspecified, they default to
A, B, C, etc.  Most routines requiring specification of a particle type index
(0, 1, 2, etc.) will also accept a particle type name.  You can use the
:py:func:`~paccs.crystal.Cell.index` and
:py:func:`~paccs.crystal.Cell.name` methods to convert between the two
types of specifiers.

Cell geometry
^^^^^^^^^^^^^

A few routines are useful for acquiring information about the geometry of the
cell itself.  You can calculate the volume (or area) of a cell, surface area
(or perimeter) of a cell, its distortion factor, or normals to its faces.  The
distortion factor is a quantity which is equal to 1 when the cell is a perfect
cube, and greater than 1 otherwise.

.. code-block:: python

   >>> cell3 = crystal.Cell(
   ...     numpy.array([[1, 0.1, 0.2], [0.3, 1, 0.4], [0.5, 0.6, 1]]), [])
   >>> cell3
   <paccs.crystal.Cell in 3D>
   >>> cell3.enclosed
   0.686
   >>> cell3.surface
   5.8516869995258265
   >>> cell3.distortion_factor
   1.6167426497232626
   >>> cell3.normals
   array([[ 0.76, -0.1 , -0.32],
          [ 0.02,  0.9 , -0.55],
          [-0.16, -0.34,  0.97]])

Particle pair measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^

A cell object allows measurements of particle pair separation distances to be
made.  All objects keep an automatically maintained internal cache of
measurements which allows for fast repeated calls to measurement routines.
However, creating a new cell with the same properties as an existing one will
not copy the cache; if this is a concern and you wish to exercise some control
over this cache, see :py:func:`~paccs.crystal.Cell.measure_to`,
:py:func:`~paccs.crystal.Cell.read_rdf`, and
:py:func:`~paccs.crystal.Cell.write_rdf`.

To make use of the measurement capabilities, you can retrieve a discrete and
unnormalized (no division by the factor of :math:`4\pi\rho r^2` in 3D or :math:`2\pi\rho r` in 2D, nor by the number of times measured, i.e., number of "centers") RDF::

   >>> cell2.rdf("X", "Y", 4)
   {0.5: 4,
    1.1180339887498949: 8,
    1.5: 4,
    1.8027756377319946: 8,
    2.0615528128088303: 8,
    2.5: 12,
    2.6925824035672519: 8,
    3.0413812651491097: 8,
    3.2015621187164243: 8,
    3.3541019662496847: 8,
    3.5: 4,
    3.640054944640259: 8,
    3.905124837953327: 8}

For convenience, you can retrieve minimum contact distances as well::

   >>> cell2.contact("X", "X")
   1.0
   >>> cell2.contact("Y", "Y")
   0.70710678118654757
   >>> cell2.contact("X", "Y")
   0.5

Finally, it is possible to retrieve a scale factor based on particle radii.
This uses particle contacts to calculate a conversion ratio between the units
of the radii and the units of the cell coordinates:

* If the cell coordinates have physically meaningful units and the radii are
  relative, multiply the radii by the scale factor to convert them to the
  same units as the cell coordinates.

* If the radii have physically meaningful units and the cell coordinates are
  relative, divide the coordinates by the scale factor to convert them to the
  same units as the radii.

.. code-block:: python

   >>> cell2.scale_factor((1, 1))
   0.25
   >>> cell2.scale_factor((1, 10))
   0.035355339059327376
   >>> cell2.scale_factor((5, 1))
   0.083333333333333329

.. warning::
   Assumptions are made during contact calculation which expect all particles
   in a cell to be within its periodic box.  If this is not the case, results
   may be incorrect (particles may be missed).  Use
   :py:func:`~paccs.crystal.CellTools.wrap` before calculating contacts
   if it is suspected that particles may not obey periodic boundary conditions.

Additional manipulation tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a number of useful tools in the :py:class:`~paccs.crystal.CellTools`
class for manipulating cells.  The following list offers a brief summary of each:

* :py:func:`~paccs.crystal.CellTools.similar` compares two cells and
  determines if their parameters are equal to within a given tolerance.  This
  is *not* a structural similarity comparison.

* :py:func:`~paccs.crystal.CellTools.identical` compares two cells and
  determines if their parameters are equal exactly.  Be aware of floating-point
  rounding issues when comparing cells in this way.

* :py:func:`~paccs.crystal.CellTools.rename` modifies particle type
  names without making changes to the particles themselves.

* :py:func:`~paccs.crystal.CellTools.reassign` modifies particle types.
  Deletion, reordering, and merging of particle types is permitted.

* :py:func:`~paccs.crystal.CellTools.scale` allows new vectors to be
  assigned to a cell.  It is possible to change the vectors only, or move the
  particles as well to achieve a stretching or shearing effect.

* :py:func:`~paccs.crystal.CellTools.normalize` rotates a cell such
  that its matrix of row vectors is upper triangular.

* :py:func:`~paccs.crystal.CellTools.wrap` enforces periodic boundary
  conditions by wrapping particles out of range back into the cell.

* :py:func:`~paccs.crystal.CellTools.condense` removes duplicate
  particles (or any particles of identical types within a certain cutoff).

* :py:func:`~paccs.crystal.CellTools.shift` translates particles in a
  cell such that a specific particle is in a given position.  By default, the
  first particle of the first type will be placed at the origin.

* :py:func:`~paccs.crystal.CellTools.tile` generates periodic
  primitive or supercells.  It can also be used to create cells containing all
  particles whose volumes intersect the cell (rather than just their centers).
  This can be useful for visualization.

* :py:func:`~paccs.crystal.CellTools.reduce` attempts to modify cell
  vectors to minimize the distortion factor.  It can also automatically call a
  number of other routines and is useful with its default options for a general
  cell cleanup.

The :py:class:`~paccs.crystal.CellCodecs` class contains a few routines
for reading and writing cell objects.  Although these objects should be able to
pass through :py:mod:`pickle` without any issues, you may wish to work with a
human-readable format.

* :py:func:`~paccs.crystal.CellCodecs.write_xyz` allows for XYZ export.
  Information about the cell vectors will be written to the comment line of the
  XYZ file.

* :py:func:`~paccs.crystal.CellCodecs.write_lammps` allows for export in
  the format understood by the `LAMMPS <http://lammps.sandia.gov/>`_ simulation
  package.  This is currently only available for three-dimensional cells.

* paccs provides its own format which should preserve all information
  about a cell object (other than cached contact information).  This format
  supports both reading (:py:func:`~paccs.crystal.CellCodecs.read_cell`)
  and writing (:py:func:`~paccs.crystal.CellCodecs.write_cell`).

These formats are best described by example::

   >>> cell = crystal.Cell(2.0 * numpy.eye(3), [
   ...     numpy.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]),
   ...     numpy.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]])],
   ...     ["Na", "Cl"])
   >>> import io
   >>> f = io.StringIO(); crystal.CellCodecs.write_xyz(cell, f)
   >>> print(f.getvalue())
   8
   2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 2.0
   Na 0.0 0.0 0.0
   Na 0.0 1.0 1.0
   Na 1.0 0.0 1.0
   Na 1.0 1.0 0.0
   Cl 0.0 0.0 1.0
   Cl 0.0 1.0 0.0
   Cl 1.0 0.0 0.0
   Cl 1.0 1.0 1.0
   >>> f = io.StringIO(); crystal.CellCodecs.write_lammps(cell, f)
   >>> print(f.getvalue())
   LAMMPS

   8 atoms
   2 atom types

   0.0 2.0 xlo xhi
   0.0 2.0 ylo yhi
   0.0 2.0 zlo zhi
   0.0 0.0 0.0 xy xz yz

   Atoms

   1 1 0.0 0.0 0.0
   2 1 0.0 1.0 1.0
   3 1 1.0 0.0 1.0
   4 1 1.0 1.0 0.0
   5 2 0.0 0.0 1.0
   6 2 0.0 1.0 0.0
   7 2 1.0 0.0 0.0
   8 2 1.0 1.0 1.0
   >>> f = io.StringIO(); crystal.CellCodecs.write_cell(cell, f)
   >>> print(f.getvalue())
   3 4 4
   Na
   Cl
   2.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00
   0.0000000000000000e+00 2.0000000000000000e+00 0.0000000000000000e+00
   0.0000000000000000e+00 0.0000000000000000e+00 2.0000000000000000e+00
   0.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00
   0.0000000000000000e+00 1.0000000000000000e+00 1.0000000000000000e+00
   1.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00
   1.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00
   0.0000000000000000e+00 0.0000000000000000e+00 1.0000000000000000e+00
   0.0000000000000000e+00 1.0000000000000000e+00 0.0000000000000000e+00
   1.0000000000000000e+00 0.0000000000000000e+00 0.0000000000000000e+00
   1.0000000000000000e+00 1.0000000000000000e+00 1.0000000000000000e+00

Calculating energies
--------------------

Energies can be calculated directly from cell objects and pair potentials if
desired.  A few different options are available for specifying the pair
potentials, and there are currently two evaluators implemented.

Potential objects
^^^^^^^^^^^^^^^^^

Pair potentials are defined in the :py:mod:`~paccs.potential` module.
The :py:class:`~paccs.potential.Potential` class is the base class for
all pair potentials, and instances have the methods:

* :py:func:`~paccs.potential.Potential.evaluate`, which provides the
  energy and force for a given single separation distance.

* :py:func:`~paccs.potential.Potential.evaluate_array`, which permits
  evaluation of energy and force for many separation distances simultaneously.

Creating a :py:class:`~paccs.potential.Potential` on its own creates a
zero potential (the energy and force are zero everywhere).  Predefined
potentials include:

* :py:class:`~paccs.potential.Transform` performs horizontal and
  vertical shifts and scalings on existing potentials.

* :py:class:`~paccs.potential.Piecewise` can patch two existing
  potentials together at a given separation distance.

* :py:class:`~paccs.potential.LennardJonesType` creates Lennard-Jones
  and Lennard-Jones-like potentials.

* :py:class:`~paccs.potential.JaglaType` creates Jagla-type potentials.

You can create combinations of these predefined potentials for evaluation::

   >>> p1 = potential.LennardJonesType(n=8, s=1)
   >>> p1.evaluate(1)
   (-1.0, -1.6289264039043005e-15)
   >>> p1.evaluate_array(numpy.linspace(0.9, 1.1, 5))
   (array([ 0.34229477, -0.7923509 , -1.        , -0.90917311, -0.74562776]),
    array([  4.03973702e+01,   1.02001650e+01,  -1.62892640e-15,
            -2.95374565e+00,  -3.35965628e+00]))

Alternatively, you can define your own potentials as subclasses of
:py:class:`~paccs.potential.Potential`::

   >>> class HertzPotential(potential.Potential):
   ...     def __init__(self, sigma=1.0, epsilon=1.0):
   ...         self.__sigma = sigma
   ...         self.__epsilon = epsilon
   ...     def __pnames__(self):
   ...         return ("sigma", "epsilon")
   ...     def __reduce__(self):
   ...         return self.__class__, (self.__sigma, self.__epsilon)
   ...     def evaluate(self, r):
   ...         if r < self.__sigma:
   ...             term = 1.0 - (r / self.__sigma)
   ...             return (self.__epsilon * (term ** 2.5),
   ...                 2.5 * self.__epsilon * (term ** 1.5))
   ...         else:
   ...             return 0.0, 0.0
   ...
   >>> hertz = HertzPotential(2.0, 3.0)
   >>> hertz.evaluate_array(numpy.linspace(0, 3, 5))
   (array([ 3.        ,  0.92644853,  0.09375   ,  0.        ,  0.        ]),
    array([ 7.5       ,  3.70579413,  0.9375    ,  0.        ,  0.        ]))

This can be convenient for testing, but for best performance, it will be useful
to define your custom potential within the ``potential.pyx`` source code and
rebuild paccs.

.. note::
   Although all that is necessary for a custom potential is an ``evaluate()``
   method (as well as an ``__init__()`` method for the specification of any
   parameters), it is recommended that you define ``__reduce__()`` as shown in
   the above example.  If this method is not defined, errors will occur if you
   attempt to use the potential in a parallel processing pool (such as that
   created by the :py:mod:`~paccs.automation` module).  An implementation
   of reduce should simply return:

   1. A callable used to recreate an instance of the potential (typically the
      class).

   2. Arguments to provide to that callable (typically parameters to
      ``__init__()``) to replicate the object exactly.

   Furthermore, we recommend that the user define ``__pnames__()`` as above.  In
   combination with the ``__reduce__()`` method, this allows the user to
   reconstruct a new potential with a changed set of parameters.

   >>> pn = hertz.__pnames__()
   >>> c,p = hertz.__reduce__()
   >>> new_hertz = HertzPotential(**dict(zip(pn,p)))

   As an intermediate step, one can modifiy the dictionary of parameter names and
   values as desired.

   It is important to also see that since dicts are unordered, the initializer
   list cannot be assumed to be in any particular order.  Therefore, all ``__init__()``
   arguments should NOT be given as positional, if the user wants to take advantage
   of this.

Energy evaluation
^^^^^^^^^^^^^^^^^

There are currently two implementations of the energy evaluation code.  The
first uses the cached pairwise contact information stored within cell objects.
This is implemented in pure Python and tends to be rather slow.  The second is
a standalone implementation which is compiled to C when paccs is built.
In addition to the energy, this implementation also provides forces on all
particles in the cell.  Both implementations should return identical energies
excepting any rounding errors::

   >>> # Create cell objects
   >>> NaCl = crystal.Cell(2.0 * numpy.eye(3), [
   ...     numpy.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]),
   ...     numpy.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]])])
   >>> CsCl = crystal.Cell(2.0 * numpy.eye(3), [
   ...     numpy.array([[0, 0, 0]]), numpy.array([[1, 1, 1]])])
   >>> # Rescale them based on contact information
   >>> NaCl = crystal.CellTools.scale(NaCl, NaCl.vectors / NaCl.scale_factor((1, 1)))
   >>> CsCl = crystal.CellTools.scale(CsCl, CsCl.vectors / CsCl.scale_factor((1, 1)))
   >>> # Create potentials
   >>> potentials = {("A", "A"): potential.LennardJonesType(sigma=2, lambda_=0, s=1),
   ...     ("B", "B"): potential.LennardJonesType(sigma=2, lambda_=0, s=1),
   ...     ("A", "B"): potential.LennardJonesType(sigma=2, lambda_=1, s=1)}
   >>> # Perform the measurements using the Python implementation
   >>> crystal.CellTools.energy(NaCl, potentials, 7.5)
   -3.6318941576958617
   >>> crystal.CellTools.energy(CsCl, potentials, 7.5)
   -4.253624140888715
   >>> # Perform the measurements using the C implementation
   >>> potential._evaluate_fast(NaCl, potentials, 7.5)
   (-3.6318941576958643,
    array([ -2.94360890e-17,  -1.23327997e-18,  -3.63207728e-18,
            -1.34305544e-17,   1.00288701e-18,  -4.41812385e-18,
             1.97324795e-17,   5.24482801e-18,  -1.26038503e-18,
             1.84314369e-17,  -3.15773883e-18,  -2.30392962e-19,
            -2.01605885e-17,  -2.65691736e-18,  -3.62503632e-20,
            -1.68545726e-17,   1.27383498e-18,   2.77132637e-18,
             3.76663773e-17,   2.19222384e-18,   2.43178527e-18,
             2.01654590e-17,   2.65522330e-18,  -3.43383186e-18]))
   >>> potential._evaluate_fast(CsCl, potentials, 7.5)
   (-4.2536241408887525,
    array([ -2.97071395e-17,   3.25260652e-18,   0.00000000e+00,
             2.97071395e-17,  -3.25260652e-18,   0.00000000e+00]))

Note that the energies are nearly identical.  In this example, the particles
are placed at their equilibrium positions so forces on them are near zero.
**Finally, keep in mind that these energies are reported per particle**::

   >>> crystal.CellTools.energy(NaCl, potentials, 7.5)
   -3.6318941576958617
   >>> crystal.CellTools.energy(crystal.CellTools.tile(NaCl, (2, 2, 2)),
   ...     potentials, 7.5)
   -3.6318941576958617

Visualization
-------------

The :py:mod:`~paccs.visualization` module has two visualizers for viewing
renderings of cell objects.

* :py:func:`~paccs.visualization.cell_mayavi`, using the `Mayavi
  <http://docs.enthought.com/mayavi/mayavi/>`_ rendering package.  This package
  uses VTK and creates a display window using your window manager.

* :py:func:`~paccs.visualization.cell_plotly`, using the `Plotly
  <https://plot.ly/python/>`_ rendering package.  This package generates an
  HTML page and creates a display window using your web browser.

The ``mayavi`` and ``plotly`` packages must be installed to use the respective
renderers.  Note that at this time, the
:py:func:`~paccs.visualization.cell_mayavi` renderer has additional
features not present in the :py:func:`~paccs.visualization.cell_plotly`
renderer, including the ability to select particles and view their types and
coordinates.  Detailed information on the options accepted by each renderer is
available in the API documentation; but when using default options, the
interfaces are identical::

   >>> visualization.cell_mayavi(NaCl, (1, 1, 1), (1, 1))
   >>> visualization.cell_plotly(CsCl, (1, 2, 3), (1, 1.5), partial=True)

The first example, showing the default options, produces the following rendering:

.. image:: _static/mayavi.*
   :width: 500px

The selection box appearing around the particle in the foreground is a result
of a selection made with the mouse.  The second example produces the following
rendering:

.. image:: _static/plotly.*
   :width: 500px

Take note of the parameters provided to the renderer.

* The first tuple passed in after the cell object specifies the number of
  repeats in various directions.  In this case, the cell is three-dimensional,
  so it has three elements ``(1, 2, 3)`` indicating the number of repeats along
  the x-, y-, and z-axes.

* The second tuple passed in after the cell object specifies particle radii.
  Automatic scaling and contact detection is performed.  These are relative
  radii; the coordinates displayed along the axes are scaled according to the
  cell itself.

* The optional ``partial`` argument indicates that additional partial periodic
  images should be generated.  This is responsible for the generation of the
  additional blue particles seen in the figure.
