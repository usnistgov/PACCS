# paccs: Python Analysis of Colloidal Crystal Structures.

This is paccs, a colloidal crystal structure analysis, generation,
optimization, and visualization library written in Python.  It supports
stochastic optimization of periodic multicomponent systems in two and three
dimensions, as well as the generation of candidate structures in two dimensions
using wallpaper groups.  A module is included to permit simple parallelization
of large generation and optimization jobs, as well as the collection and
filtering of the output of these jobs into a single database.

## Getting started

Once you've obtained a copy of the repository for yourself, take note that:

* As well as an installation of Python 3, paccs requires the presence of                                                                                                                                    
  some additional packages listed in the `requirements.txt` file.  We recommend                                                                                                                             
  installing these dependencies by obtaining a distribution of                                                                                                                                              
  [Miniconda](https://conda.io/en/latest/miniconda.html) and using                                                                                                                                          
  `conda install ...`; note that [python-constraint](https://labix.org/python-constraint) is unavailable through                                                                                              `conda` and must be installed with [pip](https://pypi.python.org/pypi/pip).

~~~
    $ cat requirements.txt
        cython
        mayavi
        numpy
        plotly
        python-constraint
        scikit-learn
        scipy
        sphinx
        sphinx_rtd_theme
~~~

* It is recommended that you use the latest version of Python 3 as well as the
  latest versions of all of the requirements.  You can leave out one or both of
  `mayavi` and `plotly` if you don't want to use the visualization frontends
  enabled when they are present.

* Building the paccs package can be accomplished by executing the shell script
  `install` in the package root directory.  With no options, the package will
  be built and tested automatically.  A few options are available to skip parts
  of the build or control build options for testing purposes; see `install -h`
  for a list.

~~~
    $ ./install -h
        Accepted options:
            -c  Skip compilation
            -d  Skip documentation generation
            -u  Skip unit testing
            -p  Compile with Cython profiling information
~~~

* The compiled modules will be placed in the `lib` directory.  This is the
  directory you should add to your `PYTHONPATH` to use paccs.

To start working, you can either `import paccs` or `from paccs import *` once
the libraries are in your path, or use the script `bin/paccssh`.  This script
will find and load paccs and then start a Python interactive session (or an
IPython interactive session if IPython is available).

## Documentation

The build process automatically creates HTML documentation inside
`docs/build/html`.  Information on the paccs API, as well as how to use
paccs to accomplish various common tasks, is available here.

## Other information

Please see the `LICENSE` file for package license information.
