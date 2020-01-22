.. _mi_miscfeatures:

Miscellaneous features
======================

This is a short list of any miscellaneous features that fail to fit anywhere
else in the documentation.

* There are a few utilities which live in ``/bin/`` and may be useful to add to
  your ``PATH`` if you desire.

  * ``paccssh``: Launches an interactive Python shell (or an interactive IPython
    shell if IPython is available) with paccs loaded.

  * ``paccscell``: Loads .cell files and performs a number of simple operations
    on them, including visualization.

  To get help for these utilities, pass the ``-h`` or ``--help`` flags to them.
  A brief list of options will be produced.

* If you wish to export an optimization trajectory to look at moves'
  performance or simply produce an animation of the system in operation, pass a
  ``_DEBUG_XYZ_PATH`` to :py:func:`paccs.minimization.optimize`.  This
  will create or append to the specified XYZ file on every evaluation of the
  optimizer objective function.
