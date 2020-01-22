Notes for maintainers
=====================

This document contains some information on the internal layout of the
paccs code for future reference.  This information is subject to change
and will (hopefully) be updated with the code itself.

General layout
--------------

* ``__init__.py``: contains a list of all other modules.  Be sure to
  update this when new modules are added, or all sorts of problems will start
  to occur.

* ``version.py``: contains a version string.  This will automatically
  be read by various parts of the code.  If you wish to update the version of
  the package, do so here.  If you wish to use a version string in the code,
  reference this module instead of hard-coding it.

* ``crystal.pyx``: periodic cell manipulation tools.  Does contain basic
  functionality for measuring RDFs and energies.  Many tools for working with
  Cell objects.  There are a few file exporters and one importer.

  Most of the routines here have been tested for cells of dimensions 2 and 3
  but should in theory work with any number of dimensions (not that this will
  ever be useful).

* ``potential.pyx``: potential functions and fast measurement algorithm.

* ``visualization.pyx``: cell object visualization.  The geometry generation
  (which creates the points on the surfaces of spheres) will always work.  Each
  exposed routine connects to a different rendering package and will only
  function if that package is installed.

* ``minimization.pyx``: the generation and optimization algorithms and their
  supporting routines.  This contains the core functionality of the package,
  whereas the ``crystal`` module provides an underlying framework for it.

* ``automation.py``: the multiprocessing framework.  This is somewhat complex
  as it needs to (correctly) perform asynchronous I/O.

* ``similarity.pyx``: similarity metrics for comparing cells and a filtering
  routine to condense a large list of them.

* ``enum_config.pyx``: a module to enumerate all combinatorial realizations of 
  crystals found to be possible for a given stoichiometry and lattices. This is
  used during automation by the master process to enumerate the specific 
  ones desired to be subsequently optimized.

Notes
-----

This section contains lists of some things that you should be aware of while
editing the code, as well as explanations for some oddities that you might
encounter.  There are also some suggestions for improvements.

General matters
^^^^^^^^^^^^^^^

* ``.pyx`` files will be Cythonized while ``.py`` files will not.  All code
  must use Python 3 syntax.  Cython will be invoked with the
  ``language_level`` parameter set to ``3``.

* Tests for some parts of code test for trivial things which would be extremely
  unlikely to fail, but fail to test for critical parts which might break and
  cause problems.  This could be improved in some places, but in others it is
  because a proper test is not feasible.  For instance, it is easy to check that
  a random move does not crash the minimizer, but impossible to check that it
  actually functions perfectly as desired without generating a trajectory and
  performing complicated analysis on it (or visualizing it).

The ``crystal`` module
^^^^^^^^^^^^^^^^^^^^^^

* Many functions in ``crystal.CellTools`` might be better suited as instance
  methods in ``crystal.Cell``, but the rationale behind the initial layout of
  this module was to keep special cell utilities separate from basic
  functionality that directly manipulates class attributes.

* For performance, it would probably be best for this class to be Cythonized.
  A basic Cythonization would probably break all ``@property`` functions.  A
  more thorough Cythonization would involve pulling out all of the old
  ``collections.Counter``-based RDF operations and replacing them with calls
  into ``_evaluate_fast``, as well as converting all of the code called from
  ``CellTools.reduce``.  A lot of it relies on nested comprehensions.

The ``potential`` module
^^^^^^^^^^^^^^^^^^^^^^^^

* Potential objects are ``cdef`` classes.  To define a custom one, override
  ``evaluate``.  This function should expect to operate on scalars, not vectors.

* There should be a better way to create custom potentials.  Users could override
  this class from within Python but the evaluation will become incredibly slow
  due to the function call back into the interpreter.  Use of the ``cpdef``
  method allows the entire evaluation call (besides the initial setup) to run
  in pure C.

* The ``_evaluate_fast`` routine probably belongs somewhere else (perhaps in
  the ``crystal`` module with the actual energy and RDF interface routines).
  However there is a Cython bug that causes problems when ``evaluate`` returns
  a ``tuple`` of two ``doubles`` that Cython optimizes to a ``struct`` and then
  can't seem to export.  This may or may not be the result of a known Cython
  issue, although there is probably some sort of a fix for it.

* If you need to make modifications to ``_evaluate_fast``, be sure to check
  ``/sources/Cython/paccs/potential.html`` to make sure that your
  changes haven't introduced any Python API calls into the middle of the nested
  evaluation loops.  These calls are best made above the loops in the setup
  portion of the routine.  If this is unfeasible, try exchanging loop order to
  keep your call in as few nested loops as possible.

The ``visualization`` module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The Mayavi renderer should probably write selection information to the
  graphical window instead of the console.  It might also benefit from actual
  mouse interaction with the particles (being able to move them around, etc.)
  The feasibility of this is unknown.

* The Plotly renderer does not support the sort of interaction available in the
  Mayavi renderer at all.  This would probably be very difficult and might
  involve Javascript code generation from within Python, unless Plotly updates
  their API.

The ``minimization`` module
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The ``generate_wallpaper`` routine is very large, even though two routines
  have been extracted from it (``tile_wallpaper`` and
  ``_solve_integer_problem``).  There does not seem to be much refactoring
  that can be done here without adding unnecessary complexity.

* The ``optimize`` routine is also very large and has an unmaintainable number
  of parameters.  This could probably be resolved by pulling each of the random
  moves out into their own routines.  This would allow better extensibility
  (addition by the user of a custom random move).  A way to pass parameters
  into these routines, and a way to provide them with the necessary current
  state of ``optimize`` would need to be implemented.

  * At the moment, adding a move requires adding a probability kwarg for it
    (and any other options), adding it in the probability normalization
    section, adding it to a list of choices, and then adding the code under the
    correct choice index in the ``StepTaker``.

* The two main routines use ``log(level, message)`` to report status,
  never a direct ``print``.

The ``automation`` module
^^^^^^^^^^^^^^^^^^^^^^^^^

* This cannot be Cythonized or ``multiprocessing`` will break.  Everything
  passing through here must be picklable or ``multiprocessing`` will break.
  This means that ``cdef`` classes must have ``__reduce__`` defined.

* The order in which things are set up is very important.  Various locks of
  different types have to be created.

  * The ``db_lock`` for writing to the database lives in the main process only.
    This is the only process which writes to disk, although it does so
    asynchronously from the generation and task spawning routine.
  
  * The ``log_lock`` is a special managed lock so that it can be passed to
    worker processes.  Unlike the ``db_lock``, using this improperly won't
    cause any harm to the real process output (the database).  It will,
    however, cause log messages of different processors to become mixed up.

* The ``_MAGIC_BEGIN`` and ``_MAGIC_END`` constants are just written at the
  beginning and end of every block in the database.  If something goes very
  wrong, you could theoretically go in to a database with a hex editor and
  search for these to pull out good blocks.  Right now recovery like this is
  not automatic; these values are just used to check that blocks have been
  written fully.

* The ``CellProcessor``, ``ScalingProcessor``, and ``TilingProcessor`` objects
  are all very nice self-contained convenience objects to allow things to be
  preprocessed and postprocessed.  The ``AutoFilteringProcessor`` is a kludge
  to try to insert easily applied filtering into the optimization pipeline
  without making the API ugly.  This could be done better if the entire
  pipeline were rewritten to make it more general.

The ``enum_config`` module
^^^^^^^^^^^^^^^^^^^^^^^^^^

* This contains a rather "manual" way of recursively enumerating combinatoric
  sets.  However, :py:mod:`itertools` contains certain functions which might lead
  to an improved (presumably faster) way doing this enumeration in the future.
  However, the speed is currently plenty adequate so this has not been revisited
  at this time.

Flowcharts
----------

The following flowcharts describe three main areas of the paccs code
that execute when conducting an automated optimization run.

Multiprocessing (the :py:mod:`paccs.automation` module)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/multiprocessing.*
   :width: 100%
   :height: 500px

Guess generation (the :py:func:`paccs.minimization.generate_wallpaper` routine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/generation.*
   :width: 100%
   :height: 500px

Basin-hopping (the :py:func:`paccs.minimization.optimize` routine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/basinhopping.*
   :width: 100%
   :height: 500px
