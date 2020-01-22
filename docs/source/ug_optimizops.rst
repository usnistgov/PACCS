.. _ug_optimizops:

Solving structure optimization problems
=======================================

This guide discusses features in the :py:mod:`~paccs.minimization`,
:py:mod:`~paccs.automation`, and :py:mod:`~paccs.similarity`
modules.

Performing automated runs
-------------------------

The annotated example script below describes most of the basic features that
can be used to perform a run using the :py:mod:`~paccs.automation`
module.  Some of the additional options available, as well as what can be done
with the results of an automated run, are described in following sections of
this guide.

.. literalinclude:: _static/run.py

This script may take a long time to execute depending on the number of cores
available on your machine.  For testing purposes, it may be helpful to adjust
``sample_count``, ``sample_groups``, ``niter``, ``niter_success``, and/or
``count`` to decrease job completion time.  Logging output will be sent to
``STDERR``; you can redirect it to a file if you wish.  If you are running from
a shell, sending Ctrl+C should stop the master process and all worker processes.

Whether the script terminates normally or abnormally, a database with the
specified filename will be created.  paccs will do its best to not
corrupt the database even if processes are killed during a run.  Information on
extracting and filtering the data in the file is available later on in this
guide.

Parameters and additional customizations
----------------------------------------

The example script above shows a number of the essential options needed to
customize an automated run.  However, there are a large number of additional
parameters that can be used to modify runs.  The API documentation is the
authoritative source for these, but this section intends to provide an overview
of these options and when they might be useful.

The :py:func:`~paccs.wallpaper.generate_wallpaper` routine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consult the API documentation for a detailed description of the parameters.
Keep in mind the following points when using this routine in an automated run:

* Do not specify the ``log_file`` option; it will automatically be specified
  so that log output can be captured and stored in the database.

* Ensure that the ``weighting_function``, if you are specifying a custom one,
  is defined as a normal function (using ``def``) or other callable object at
  the global level in your run script.  It will be pickled when the problem
  description is sent to worker processes, and these processes must be able to
  locate it.  Using a nested function, nested ``lambda``, or an instance method
  may cause issues.

* For debugging, you may wish to use the ``count_configurations`` option to see
  the number of configurations you can generate using a given set of
  parameters.  This is useful in an interactive session as it does not try to
  generate every possible configuration: it only calculates how many exist.

The :py:func:`~paccs.minimization.optimize` routine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consult the API documentation for a detailed description of the parameters.
Keep in mind the following points when using this routine in an automated run:

* You must specify ``potentials`` and ``distance``.  Never specify ``cell`` or
  ``log_file`` in an automated run; this is done automatically when a worker
  process is launched.

* The ``enclosed_bounds`` and ``surface_bounds`` options have been selected to
  avoid extremely distorted cells from being evaluated by the local optimizer.
  Defaults have been chosen that should be reasonable for cells with particle
  separations on the order of 1 (as may be the case if your problem is
  specified in reduced units).  If this is not the case for your problem, you
  may need to change these defaults.

* For reference, the following options correspond to the different moves
  available:

  * Exchange moves: ``exchange_move``, ``exchange_select``.
  * Arbitrary vector moves: ``vector_move``, ``vector_select``,
    ``vector_shear``, ``vector_factor``
  * Uniform scale moves: ``scale_move``, ``scale_factor``
  * Particle displacement moves: ``atom_move``, ``atom_select``
  * Cluster moves: ``cluster_move``, ``cluster_factor``

  The ``_move`` parameters specify probabilities of each move.  If these
  do not sum to 1, they will automatically be normalized.

* The ``initial_kwargs`` and ``final_kwargs`` options accept those options
  accepted by :py:func:`scipy.optimize.minimize`.  You should only attempt to
  specify ``method``, ``tol``, or ``options``; others may be ignored or break
  the minimizer.  Consult the SciPy documentation for valid forms of
  ``options`` for different values of ``method`` accepted.

* The ``basin_kwargs`` option accepts those options accepted by
  :py:func:`scipy.optimize.basinhopping`.  You should only attempt to specify
  ``T``, ``interval``, ``niter``, or ``niter_success``.  You may also specify
  ``minimizer_kwargs`` with the same cautions as above.  Other options may be
  ignored or break the minimizer.

Cell preprocessors and postprocessors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are a few processors available in the :py:mod:`~paccs.automation`
module.  These processors perform arbitrary actions on cells entering and
exiting the basin-hopping optimizer and are executed inside worker processes.
The ones available are described above in the example script.  However it
should be possible to create your own custom processor by subclassing
:py:class:`~paccs.automation.CellProcessor`::

   >>> class MyProcessor(automation.CellProcessor):
   ...     def __init__(self, my_parameter):
   ...         self.__my_parameter = my_parameter
   ...     def __call__(self, cell):
   ...         print("Got a cell with d.f. {}".format(cell.distortion_factor))
   ...         return crystal.CellTools.tile(cell, [1] * cell.dimensions)
   ...

This particular example is useless as it does nothing to the cells passing
through it (additionally, you should not ``print`` from a processor since it
will be executed inside a worker process and the message could be mixed up with
other messages from other workers).  However, it illustrates how to create a
custom processor.  You can perform any actions to the cell inside the
``__call__``.  If you are using such a processor from an automated run script,
be sure that the class is defined at the global level so that worker processes
can use its pickled representation.

There are two special processors, the first of which is the
:py:class:`~paccs.automation.AutoFilteringProcessor`.  Unlike normal
processors, which simply modify cells entering and leaving the optimizer, this
filtering processor performs a few actions at once:

* It removes itself from the list of preprocessors to be executed by worker
  processes (specifying this processor as a postprocessor will perform no
  action whatsoever).

* It breaks up all of the jobs represented by tuples of dictionaries such that
  there exists one job per wallpaper group.  As a result, if no overrides are
  specified for wallpaper groups, then the number of jobs that will end up
  being performed will be 17 times greater than the number of cells (count)
  allowed to pass the filtering stage.

* It inserts a call to filter all cells coming out of the generator using
  :py:func:`~paccs.minimization.filter`.  The ``potentials`` and
  ``distance`` options will be automatically extracted from keyword arguments
  dispatched to the optimizer.  The ``radii`` option will be extracted from a
  :py:class:`~paccs.automation.ScalingProcessor` present in the
  preprocessing chain.  The ``histogram_callback`` option will be overridden
  such that the energies can be stored in the results database.  Finally, the
  ``count`` and ``similarity_metric`` options are specified as arguments when
  the :py:class:`~paccs.automation.AutoFilteringProcessor` is created.

Note that similarity, if a ``similarity_metric`` is specified, is assessed on the rescaled cells if a :py:class:`~paccs.automation.ScalingProcessor` is provided.

The second, non-standard preprocessor is the :py:class:`~paccs.automation.CellDensityProcessor`.  This performs no direct action on any cell, but instead calculates the ``enclosed_bounds`` for a cell during optimization.  These extensive "area" bounds cannot be fairly compared across cells if they differ in the total number of atoms present.  This processor computes the ``enclosed_bounds`` for each basin hopping run based on the density bounds the :py:class:`~paccs.automation.CellDensityProcessor` is initialized with.

* This can be particularly important as even moderately long runs tends to "explode" as the box is scaled to minimize overlaps that inevitably occur during perturbation of atom coordinates.  Therefore, **this processor is  recommended as a best-practice** if bounds for the cells can be reasonably estimated a priori.

Reading databases and filtering results
---------------------------------------

Once you have completed a run, the
:py:class:`~paccs.automation.ResultsDatabase` class can be used to read
the database generated.  For this example, we will suppose that the example
script has been permitted to complete and generate a file ``output.db``::

   >>> db = automation.ResultsDatabase("output.db")

Database files are binary files containing a sequence of ``gzip`` compressed
pickled objects.  Creating a
:py:class:`~paccs.automation.ResultsDatabase` does not load the entire
contents of a database into memory, only a table of object locations.  When
objects are requested, they are read from the file, decompressed and
reconstituted, and then cached for fast retrieval later on.

Working with database objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can interact with the database at the object level, although this may not
be particularly useful::

   >>> db.block_count
   187
   >>> db.read_block(20)
   _GenerateResult(index=15, log=..., accepted_energies=[...], rejected_energies=[...])

You can also iterate over all objects in the database, structure generation
results alone, or structure optimization results alone::

   >>> # Retrieve all results and index a single result (this is equivalent to
   ... # but less efficient than invoking db.read_block to retrieve only one)
   ... list(db.results)[20]
   _GenerateResult(index=15, log=..., accepted_energies=[...], rejected_energies=[...])
   >>> # Retrieve only the results from generation, one at a time
   ... results = db.generate_results
   >>> next(results)
   _GenerateResult(index=0, log=..., accepted_energies=[...], rejected_energies=[...])
   >>> next(results)
   _GenerateResult(index=1, log=..., accepted_energies=[...], rejected_energies=[...])
   >>> # Retrieve only the results from optimization, one at a time
   ... results = db.optimize_results
   >>> next(results)
   _OptimizeResult(
       success=True,
       indices=(0, 1),
       symmetry_group=<paccs.wallpaper.WallpaperGroup as 1/p1/o tiling 1 parallelogram>,
       initial_guess=(<paccs.crystal.Cell in 2D with 4 A, 8 B>, -0.35242293287528353),
       candidates=[
           (<paccs.crystal.Cell in 2D with 4 A, 8 B>,
            -1.6316604070719611,
            31.549055099487305,
            23.546239),
           ...,
           (<paccs.crystal.Cell in 2D with 4 A, 8 B>,
            -1.778062840298733,
            53.967509031295776,
            40.867676)],
       log=...,
       start_time=1501794289.876155,
       end_time=1501794343.89427,
       processor_time=40.871455)
   >>> next(results)
   _OptimizeResult(success=True, indices=(0, 0), ...)

.. note::
   Iterating over a generator created by a
   :py:class:`~paccs.automation.ResultsDatabase` does not permanently
   exhaust an internal generator within the database.  These generators are
   returned from properties which create new instances every time that they are
   referenced.  Thus, accessing ``db.results`` twice creates two separate
   independent generator objects, and calling ``db.results.next()`` twice will
   always yield the first result.

The output above has been formatted in a way that highlights a few important details:

* The order in which objects are returned is the order in which they were saved
  to the database.  This order is associated with the times at which the
  generation and optimization tasks finished, not the times at which they
  started.  The ``index`` values of generation result objects and the
  ``indices`` values of optimization result objects will not necessarily be in
  order.  Furthermore, generation and optimization results may be mixed
  together in the sequence of all results.

* The ``log`` fields have values because logging information was requested in
  the example run script.  Likewise, the ``accepted_energies`` and
  ``rejected_energies`` fields have values because an
  :py:class:`~paccs.automation.AutoFilteringProcessor` was present in
  the preprocessing chain.  The result objects are
  :py:func:`collections.namedtuple` instances, so all fields will always be
  present.  If a feature is disabled, its corresponding field in a result
  object will still be accessible even though its value will be ``None``.

* In :py:class:`~paccs.automation.OptimizeResult` objects, the
  ``initial_guess`` field contains a cell and its energy per particle.  Items in the
  ``candidates`` list contain cells, their energies, the wall time relative to
  the start of the run at which they became candidates, and the process time
  relative to the start of the run at which they became candidates.  Note that
  these times may differ significantly during the start of a large run due to
  simultaneous execution of a generation operation in the main process and
  optimization operations in worker processes.

For additional information, refer to the reference documentation for
:py:class:`~paccs.automation.GenerateResult` and
:py:class:`~paccs.automation.OptimizeResult`.

Working with cells
^^^^^^^^^^^^^^^^^^

Convenience generators can be created to access the cell objects within a
database without extracting them from each optimization result object.  If you
are not interested in the generation or optimization parameters leading to the
creation of the cells, these may be useful::

   >>> len(list(db.cells_in))
   170
   >>> len(list(db.cells_out))
   1536
   >>> db.cells_in.next()
   (<paccs.crystal.Cell in 2D with 4 A, 8 B>, -0.35242293287528353)
   >>> db.cells_out.next()
   (<paccs.crystal.Cell in 2D with 4 A, 8 B>, -1.6316604070719611)

Note that timestamp information is not provided by
:py:func:`~paccs.automation.ResultsDatabase.cells_out`.  This data must
be extracted directly from optimization result objects if it is desired.

The tools in the :py:mod:`~paccs.similarity` module can be used to
create a unique collection of cells as determined by some similarity measure or
`metric <https://en.wikipedia.org/wiki/Similarity_measure>`__.  A
few optional are available, not all of which are true `metrics <https://en.wikipedia.org/wiki/Metric_(mathematics)>`_:

* The :py:class:`~paccs.similarity.Energy` measure, which compares
  energies directly to within a given tolerance.  Usually this uses energy per particle.

* The :py:class:`~paccs.similarity.Hybrid` function, which allows the
  short-circuited chaining of multiple measures/metrics.

* The :py:class:`~paccs.similarity.Histogram` measure, which compares
  cells based on smoothed histograms created from their radial distribution
  functions.  A norm can be specified to control how the `cosine similarity
  metrics <https://en.wikipedia.org/wiki/Cosine_similarity>`_ from each particle
  type pair are combined into a single value. Because of this, this is not
  guaranteed to always provide a unique representation of configuration, or be a matric,
  but it is usually fairly good.

* The :py:class:`~paccs.similarity.CosHistogram` metric, which computes
  the cosine similarity metric using all i-j pair types implicitly, and weights
  the contribution by the frequency of the i-j particles. This is described in
  detail `here <http://aip.scitation.org/doi/abs/10.1063/1.3079326>`_.

* The :py:class:`~paccs.similarity.OVMeasure` measure.

* The :py:class:`~paccs.similarity.PartialRDF` metric.

The user should read in more detail in :py:class:`~paccs.SimilarityMetric` about whether the function or "metric" being used truly represents a mathematical metric or not, on a case-by-case basis. The following example illustrates the use of the
:py:class:`~paccs.similarity.Histogram` function::

   >>> cells = list(db.cells_out)
   >>> len(cells)
   1536
   >>> metric = similarity.Histogram(6, 0.25, 0.9, norm=similarity.Minkowski(1.5))
   >>> reduced = similarity.reduce(cells, metric)
   >>> len(reduced)
   15
   >>> reduced
   [(<paccs.crystal.Cell in 2D with 4 A, 8 B>,
     -1.7780628403537868),
    ...,
    (<paccs.crystal.Cell in 2D with 24 A, 48 B>,
     0.1374714237366289)]

In this case, the "metric" object created generates radial distribution histograms
with a bin width of 0.25 out to a distance of 6.00, for each i-j pair of particle types.
The cosine similarity *metric* for each pair type in a system containing :math:`n` particle
types is combined with all other pair types using an
:math:`L^p` `norm <https://en.wikipedia.org/wiki/Minkowski_distance>`_ scaled to give a similarity *measure* with values
between 0 and 1 (:math:`p=3/2` in this example).  The result is a single scalar value
characterizing similarity.

.. math::
   S(c,c')={\left[\frac{2}{n\left(n+1\right)}\sum_{i=1}^n\sum_{j=i}^n{S_{ij}(c,c')}^p\right]}^{1/p}

where :math:`S_{ij}(c,c')` for two cells designated :math:`c` and :math:`c'` is:

.. math::
   S_{ij}(c,c')=\frac{\sum_{k=1}^mg_{ij}(r_k)g'_{ij}(r_k)}{\sqrt{\sum_{k=1}^m{g_{ij}(r_k)}^2}\sqrt{\sum_{k=1}^m{g'_{ij}(r_k)}^2}}

Calling :py:func:`~paccs.similarity.reduce` returns a list of cells and
their energies such that no pair of cells within the list should be considered
similar according to the metric.  The list will contain the set of cells with
the lowest possible energies that meet this criterion.
