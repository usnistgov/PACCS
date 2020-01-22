Installing and building paccs
====================================

The ``README`` file in the package root directory gives an overview of the
installation process.  This documentation page provides more in-depth
information.

Packages required
-----------------

paccs requires a working installation of Python 3 (3.7.4 or later is recommended) as well as some
additional packages.  All packages required to use all features of paccs
are listed in the ``requirements.txt`` file.  The following packages should be
installed in order for the software to run:

* ``numpy``
* ``python-constraint``
* ``scikit-learn``
* ``scipy``

You will also need the following packages to build it from source:

* ``cython``
* ``sphinx``
* ``sphinx_rtd_theme``

Install one or both of the following to use the built-in visualization
feature.  Using Mayavi is recommended as this renderer has additional features
not currently available with the Plotly renderer.  The remainder of the package
apart from the visualizer will function normally if neither of these are
available.

* ``mayavi``
* ``plotly``

If you are using the Miniconda Python distribution, be aware that the
``python-constraint`` package cannot be installed using ``conda install``;
you must use ``pip install`` instead.

Finally, the DNACC packge is also required to use the :py:class:`~paccs.potential.DNACC` potential class.
This is optional and the code will run and pass unittests even if it is not installed.
DNACC can be downloaded `here <https://github.com/patvarilly/DNACC>`_ and any use of this
functionality **must** cite:

* `Varilly, Angioletti-Uberti, Mognetti and Frenkel, "A general theory of DNA-mediated and other valence-limited colloidal interactions" <https://doi.org/10.1063/1.4748100>`_, *J. Chem. Phys.* **137**, 094108 (2012).

See included instructions on installation, and be sure that the resulting
installation is in your $PYTHONPATH.

Installation
------------

Once you have acquired the source and set up your environment, run the
``install`` shell script to build paccs.  In addition to generation of
binaries, documentation will also be built automatically and the test suite
will be executed.  This may take several minutes to complete.

General notes
^^^^^^^^^^^^^

* To speed up repeated compilations, ``install`` will automatically detect the
  presence of ``ccache`` and use it if it is available.

* Converted source code as well as Cython conversion reports will be placed in
  ``/sources/Cython/``.  Temporary files generated during the build will be
  placed in ``/temp/``.  The compiled libraries can be found within a platform
  subdirectory under ``/build/`` to which ``/lib/`` will automatically be
  symbolically linked.  Finally, ``/docs/build/html/`` will contain the
  documentation.

* When the build script is running, a temporary file
  ``.paccs.buildinprogress`` will be created to prevent multiple builds
  from being launched simultaneously.  The build process can be interrupted
  using Ctrl+C at any time, which should normally cause this file to be
  deleted.  Under certain circumstances, however, the shell script will fail to
  capture this interrupt and an error message will occur on subsequent builds.
  If this occurs, it is safe to delete the file as long as no builds are
  actually in progress.

Customizing the build
^^^^^^^^^^^^^^^^^^^^^

* The ``install`` script accepts a few different options to skip certain
  portions of the build process.  Use ``install -h`` to get a list of these
  options.

* By default, the API documentation will only include items designed to be
  utilized from outside of paccs.  To document all members, ``echo 1 >
  .sphinx_document_all`` in the package root directory.  The documentation
  build step will look for this file and read the flag.  The absence of the
  file or the presence of ``0`` instead of ``1`` will yield the default
  behavior.

* If you wish to run tests manually, use ``runtests`` in the ``/tests/``
  directory or invoke ``unittest`` on the test modules directly.  A few other
  executable scripts are available in this directory, namely ``randeval``,
  ``minimization.py``, and ``visualization.py``.  These perform additional
  tests which are not suitable for an automated build process.

Getting started
---------------

Now that paccs is installed, you may wish to do the following:

* Add the ``/lib/`` directory to your ``PYTHONPATH``, or append it to
  ``sys.path`` in scripts in which you wish to use paccs.  This will
  allow you to import the modules.  Using ``import paccs`` or ``from
  paccs import *`` will load all modules; of course, you can also load
  individual ones as desired.

* Add the ``/bin/`` directory to your ``PATH``, if you want easy access to a
  :ref:`few useful tools <mi_miscfeatures>`.

There are two guides available to help you get started using paccs:

1. :ref:`ug_crystalops`: discusses some of the basic features available to
   perform analysis on periodic cells.

2. :ref:`ug_optimizops`: discusses how to set up an automated optimization run
   and process the output data.  Various useful parameters are described here.

For a complete reference, consult the :ref:`paccs API documentation
<apidoc>`.
