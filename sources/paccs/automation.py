"""
Provides an interface for performing semi-automated parallel runs of the generation and
optimization algorithms.  Allows for the automatic assembly and processing of a database
of candidate structures.
"""

from . import crystal
from . import potential
from . import minimization
from . import wallpaper
from . import version
import collections
import fcntl
import gzip
import io
import multiprocessing
import numpy
import os
import pickle
import struct
import sys
import time
import traceback
import warnings

# Magic values at the beginning and end of each data block
# Used to try recovery in the case of a catastrophic failure
_MAGIC_BEGIN = b"/\xdc\x1a\x1c\xbe\x97\xf6\x1c\xa1\xb6\x83&&4\x80\xcc"
_MAGIC_END = b"#fW}\xe7\x16\r\xadR\xafAF\xa7\xdfX\xec"

def _print(log_lock, *args, **kwargs):
    """
    Calls the builtin print function to print to the standard
    error stream with the guarantee that no more than one process
    at a time will ever be able to do so.

    Parameters
    ----------
    log_lock : multiprocessing.managers.BaseProxy
        A proxy to a shareable lock.
    args : tuple
        Positional arguments to be passed to :py:func:`print()`.
    kwargs : dict(str, object)
        Keyword arguments to be passed to :py:func:`print()`.
    """

    # Always print to the standard error stream
    kwargs["file"] = sys.stderr

    # All operations with the stream must occur while the lock is held
    log_lock.acquire()
    print(*args, **kwargs)
    sys.stderr.flush()
    log_lock.release()

class CellProcessor:
    """
    Arbitrary cell processor.  Subclasses perform actions to
    cells.  Creating an instance of this class and using it to
    process cells will affect no changes to the cells.
    """

    def __call__(self, cell):
        return cell

class ScalingProcessor(CellProcessor):
    """
    Scales cells according to provided atomic radii.  This is a
    simple wrapper around :py:func:`crystal.CellTools.scale` and
    :py:func:`crystal.Cell.scale_factor` provided for convenience
    as it is picklable and can be dispatched to worker processes.

    Parameters
    ----------
    radii : tuple(float)
        The radii of the atom types.
    """

    def __init__(self, radii):
        self._radii = tuple(radii) # Permit extraction for AutoFilteringProcessor
    def __call__(self, cell):
        return crystal.CellTools.scale(cell, cell.vectors / cell.scale_factor(self._radii))

class TilingProcessor(CellProcessor):
    """
    Creates supercells from primitive ones.  This does NOT produce primitive
    cells from fundamental domains. This is a simple wrapper around
    :py:func:`crystal.CellTools.tile` provided for convenience
    as it is picklable and can be dispatched to worker processes.

    Parameters
    ----------
    repeats : tuple(int)
        The number of repeats to make in each direction.
    """

    def __init__(self, repeats):
        self.__repeats = tuple(repeats)
    def __call__(self, cell):
        return crystal.CellTools.tile(cell, self.__repeats)

class CellDensityProcessor(CellProcessor):
    """
    Computes ``enclosed_bounds`` on the area of a (super)cell to use during optimization.
    See :py:func:`~paccs.minimization.optimize`.
    Since the number of atoms chosen for a cell can vary, it is
    often easier to impose a density limit rather than an area limit. This is
    safe to use in conjunction with ScalingProcessor and TilingProcessor
    as the **scaling and/or tiling will occur first** to produce a candidate cell,
    then the area bounds to impose during optimization will be calculated using
    this preprocessor.  Thus, these bounds refer to the completely preprocessed
    cell.

    Calling this processor has no effect on the cells.

    Parameters
    ----------
    density_bounds : tuple(float, float)
        Minimum and maximum total number density allowable for a cell.
        Bounds will be sorted in ascending order if not provided that way.
    """

    def __init__(self, density_bounds):
        self.__density_bounds = tuple(sorted(density_bounds))
        if (len(self.__density_bounds) != 2): raise Exception('must specify density bounds as (min, max)')
        if (numpy.min(self.__density_bounds) < 0): raise Exception('minimum density bounds must be >= 0')
    def enclosed_bounds(self, cell):
        return (numpy.sum(cell.atom_counts)/self.__density_bounds[1], numpy.sum(cell.atom_counts)/self.__density_bounds[0])

class AutoFilteringProcessor(CellProcessor):
    """
    When placed anywhere within the preprocessing chain, makes modifications to
    the processing pipeline to change handling of wallpaper groups and filtering
    of cells.

    Parameters
    ----------
    count : int
        The number of generated cells that are to be processed
        for **each** selected wallpaper group.
    similarity_metric : :py:class:`~paccs.similarity.SimilarityMetric`
        Similarity metric to use to distinguish structurally similar configurations.

    Notes
    -----
    Note that the selected wallpaper groups from **sample_groups** will be detected,
    and a separate generator will be created **for each**.  If no selection of wallpaper
    groups is made, a generator will be created for each of the 17 groups.

    This is not a normal processor.  Using it in the postprocessing chain performs
    no action.  Insertion at any location in the preprocessing chain will cause cells
    to be filtered.

    * In the case that no similarity_metric is specified, only the **count** cells (or less) with the lowest energy (per atom) will be sent to optimization.

    * When a similarity_metric is provided, the lowest **count** cells (or less) in energy that are structurally unique according to this metric or measure will be optimized. Similarity is assessed on the rescaled cells if a :py:class:`ScalingProcessor` is specified.

    The **potentials** and **distance** options will be extracted from keyword arguments
    to :py:func:`paccs.minimization.optimize`.  The **radii** option will be
    extracted from a :py:class:`ScalingProcessor` in the preprocessing pipeline, if one is specified.
    """

    # Field is stored for later extraction (no operational __call__
    # because processor is handled specially by the TaskManager)
    def __init__(self, count, similarity_metric=None):
        self._count = count
        self._similarity_metric = similarity_metric

class TaskManager:
    """
    Represents a manager for a number of generation and optimization tasks.
    The manager processes any number of candidate structures by dispatching
    them to the basin-hopping optimizer.  Multiple tasks can be started in
    parallel and they will automatically be given work to do as soon as they
    become idle.

    Parameters
    ----------
    kwarg_pairs : list(dict(str, object), dict(str, object))
        A collection of pairs of keyword argument dictionaries.  Each pair
        represents one generation task which will create a number of optimization
        tasks.  The first dictionary is provided to :py:func:`paccs.wallpaper.generate_wallpaper`
        and the second dictionary to :py:func:`paccs.minimization.optimize`.
    preprocessors : list(callable)
        If specified, sends cells through a chain of preprocessing routines before
        optimization.  Each routine should accept and return a
        :py:class:`paccs.crystal.Cell` object.
    postprocessors : list(callable)
        If specified, sends cells through a chain of postprocessing routines after
        optimization.  Each routine should accept and return a
        :py:class:`paccs.crystal.Cell` object.
    worker_count : int
        If specified, will create a fixed number of workers.  By default, the
        number of workers created will be equal to the number of processors
        detected to be available.
    random_seed : int
        The seed for the Mersenne Twister random number generator that will be used
        to generate more random seeds for each generation and optimization process.
        It should be non-negative and less than :math:`2^{32}`.  If not specified,
        a randomly chosen seed will be used.  Any **random_seed** options in
        **kwarg_pairs** will be unconditionally ignored.
    """

    # Banners printed at the beginning and end of a run
    __WELCOME_MESSAGE = """
    ======================================================
    paccs: Python Analysis of Colloidal Crystal Structures
    ======================================================
        Version: {}
        Authors: Evan Pretti, Nathan A. Mahynski
        Contact: nathan.mahynski@nist.gov

        Developed at NIST.  See LICENSE file included
        with package for more information.
    """.format(version.version)
    __SUCCESS_MESSAGE = """
        Work completed successfully.
    ======================================================
    """
    __FAILURE_MESSAGE = """
        Work stopped; some failures occurred.
    ======================================================
    """

    def __init__(self, kwarg_pairs, preprocessors=[], postprocessors=[], worker_count=None, random_seed=None):
        # Make copies of keyword argument pairs
        self.__generate_kwargs, self.__optimize_kwargs = \
            zip(*([dict(kwargs) for kwargs in kwarg_pair] for kwarg_pair in kwarg_pairs))

        # Set up preprocessors and postprocessors
        self.__preprocessors = list(preprocessors)
        self.__postprocessors = list(postprocessors)

        if len([preprocessor for preprocessor in self.__preprocessors \
            if type(preprocessor) is CellDensityProcessor]) > 1:
                raise ValueError("more than one CellDensityProcessor detected")

        # Set up random seed generator (seed=None ... numpy pulls from /dev/urandom)
        self.__randomness = numpy.random.RandomState(random_seed)

        # Handle AutoFilteringProcessor
        auto_filter = [preprocessor for preprocessor in self.__preprocessors \
            if type(preprocessor) is AutoFilteringProcessor]
        if len(auto_filter) == 1:
            # Remove from the preprocessing chain
            self.__auto_filter = auto_filter[0]
            self.__preprocessors.remove(self.__auto_filter)

            # Process all work
            self.__filter_kwargs = []
            for optimize_kwargs in self.__optimize_kwargs:
                # Create new filter_kwargs
                self.__filter_kwargs.append({})

                # Handle extractable options
                self.__filter_kwargs[-1]["potentials"] = optimize_kwargs["potentials"]
                self.__filter_kwargs[-1]["distance"] = optimize_kwargs["distance"]
                self.__filter_kwargs[-1]["count"] = self.__auto_filter._count
                self.__filter_kwargs[-1]["similarity_metric"] = self.__auto_filter._similarity_metric

                # Handle radii
                scale_filter = [preprocessor for preprocessor in self.__preprocessors \
                    if type(preprocessor) is ScalingProcessor]
                if len(scale_filter) == 1:
                    self.__filter_kwargs[-1]["radii"] = scale_filter[0]._radii
                elif len(scale_filter) > 1:
                    raise ValueError("more than one ScalingProcessor detected")
                else:
                    self.__filter_kwargs[-1]["radii"] = None

            # Make copies of all kwargs per wallpaper group
            filter_kwarg_list, generate_kwarg_list, optimize_kwarg_list = [], [], []
            for pair_index, (filter_kwargs, generate_kwargs, optimize_kwargs) in \
                enumerate(zip(self.__filter_kwargs, self.__generate_kwargs, self.__optimize_kwargs)):

                # Get wallpaper groups to be processed
                groups = generate_kwargs["sample_groups"] if "sample_groups" in generate_kwargs \
                    and generate_kwargs["sample_groups"] is not None \
                    else [wallpaper.WallpaperGroup(number=index + 1) for index in range(17)]

                for group_index, group in enumerate(groups):
                    # Create new kwargs dictionaries and select just one group
                    new_filter_kwargs, new_generate_kwargs, new_optimize_kwargs = \
                        dict(filter_kwargs), dict(generate_kwargs), dict(optimize_kwargs)
                    new_generate_kwargs["sample_groups"] = [group]

                    # Add the dictionaries for this specific wallpaper group
                    filter_kwarg_list.append(new_filter_kwargs)
                    generate_kwarg_list.append(new_generate_kwargs)
                    optimize_kwarg_list.append(new_optimize_kwargs)

            self.__filter_kwargs, self.__generate_kwargs, self.__optimize_kwargs = \
                filter_kwarg_list, generate_kwarg_list, optimize_kwarg_list

        elif len(auto_filter) > 1:
            raise ValueError("more than one AutoFilteringProcessor detected")
        else:
            self.__auto_filter = None

        # Create worker pool
        self.__worker_count = multiprocessing.cpu_count() if worker_count is None else worker_count
        self.__pool = multiprocessing.Pool(self.__worker_count)

        # Keep track of whether or not work is happening
        self.__running = False

    def work(self, db_path):
        """
        Performs the assigned work by launching child processes.  Informational
        output is sent to the standard error stream.

        Parameters
        ----------
        db_path : str
            A path to a file to which output is written.  Once the work is complete,
            the database can be read with a :py:class:`ResultsDatabase`.  This file
            will be overwritten if it exists.

        Returns
        -------
        bool
            Whether or not the entire run was completely successful.  In the case
            of a failure, more information may be available in the log output or
            the results database.
        """

        try:
            # Ensure that the manager is not already running
            if self.__running:
                raise RuntimeError("the task manager is currently working")
            self.__running = True

            # Create an interprocess lock for logging
            log_manager = multiprocessing.Manager()
            log_lock = log_manager.Lock()
            # Create an intraprocess lock for file I/O
            self.__db_path = db_path
            self.__db_lock = multiprocessing.Lock()

            # Display a welcome message
            start_time, start_clock = time.time(), time.process_time()
            heading = "[manager:{}]".format(os.getpid()).ljust(20)
            _print(log_lock, TaskManager.__WELCOME_MESSAGE)
            _print(log_lock, "{} Started at {}".format(heading, time.strftime( \
                "%Y-%m-%d %H:%M:%S", time.localtime(start_time))))
            _print(log_lock, "{} Running on {} {} {} {} {}".format(heading, *os.uname()))
            _print(log_lock, "{} Detected {} CPUs available, using {} worker processes".format( \
                heading, multiprocessing.cpu_count(), self.__worker_count))

            # Check for existence of database
            if os.path.exists(self.__db_path):
                if os.path.isfile(self.__db_path):
                    _print(log_lock, "{} Database exists and will be overwritten".format(heading))
                    os.unlink(self.__db_path)
                else:
                    raise IOError("path refers to an existing object that is not a file")

            # Create tasks
            _print(log_lock, "{} Started creating tasks".format(heading))
            async_results = []
            self.__results = []
            for generate_index, (generate_kwargs, optimize_kwargs) in enumerate(zip(self.__generate_kwargs, self.__optimize_kwargs)):
                _print(log_lock, "{} Creating task set {}".format(heading, generate_index))

                # Create a generator and a list of results for this generator
                generate_kwargs["log_file"] = io.StringIO()
                generate_kwargs["random_seed"] = self.__randomness.randint(0, 2 ** 32)
                generator = wallpaper.generate_wallpaper(**generate_kwargs)
                async_results.append([])
                self.__results.append([])

                # Check for possible filtering task
                energies = [None, None]
                if self.__auto_filter is not None:
                    def filter_callback(filter_energies):
                        energies[0] = filter_energies[:self.__filter_kwargs[generate_index]["count"]]
                        energies[1] = filter_energies[self.__filter_kwargs[generate_index]["count"]:]
                    generator = minimization.filter(generator, histogram_callback=filter_callback, \
                        pool=self.__pool, **self.__filter_kwargs[generate_index])

                # Queue up all optimization tasks from the generator
                for optimize_index, (group, cell) in enumerate(generator):
                    _print(log_lock, "{} Creating task-{}-{}".format(heading, generate_index, optimize_index))
                    async_results[-1].append(self.__pool.apply_async(_work, (generate_index, optimize_index, \
                        log_lock, self.__preprocessors, self.__postprocessors, optimize_kwargs, group, cell, \
                        self.__randomness.randint(0, 2 ** 32)), callback=lambda result: self.__work_done(result, log_lock, heading)))
                    self.__results[-1].append(None)

                # Now that the filter generator is exhausted, save its log information
                self.__write_object(GenerateResult(generate_index, generate_kwargs["log_file"].getvalue(), \
                    *energies))

            _print(log_lock, "{} Finished creating tasks".format(heading))

            # Let all work finish and propagate exceptions
            self.__pool.close()
            self.__pool.join()
            _print(log_lock, "{} Finished running tasks".format(heading))
            for generate_index, generate_async_results in enumerate(async_results):
                for optimize_index, async_result in enumerate(generate_async_results):
                    # Any exceptions that occur between the catch-all try block in
                    # the _work function will get thrown up here.  This can happen
                    # if something goes wrong during pickling of _work's arguments
                    # or its return value.
                    try:
                        async_result.get()
                    except Exception:
                        self.__results[generate_index][optimize_index] = False
                        _print(log_lock, "{} Failure in task-{}-{} propagated back:\n{}".format( \
                        heading, generate_index, optimize_index, traceback.format_exc()))

            # Display termination message
            end_time, end_clock = time.time(), time.process_time()
            _print(log_lock, "{} Finished at {}".format(heading, time.strftime( \
                "%Y-%m-%d %H:%M:%S", time.localtime(end_time))))
            _print(log_lock, "{} {} seconds (wall time), {} seconds (process time) elapsed".format( \
                heading, end_time - start_time, end_clock - start_clock))
            all_success = True
            for generate_index, generate_results in enumerate(self.__results):
                for optimize_index, result in enumerate(generate_results):
                    if not result:
                        all_success = False
                        _print(log_lock, "{} Failure in task-{}-{} reported".format(heading, generate_index, optimize_index))
            _print(log_lock, TaskManager.__SUCCESS_MESSAGE if all_success else TaskManager.__FAILURE_MESSAGE)

        finally:
            # Try to shut down the pool
            try:
                self.__pool.terminate()
            except Exception:
                pass

            # Indicate that work is no longer being done
            self.__running = False

    def __work_done(self, result, log_lock, heading):
        """
        A callback triggered asynchronously when work is returned from one
        of the worker processes.

        Parameters
        ----------
        result : OptimizeResult
            The result of the work returned as a named tuple object.
        log_lock : multiprocessing.managers.BaseProxy
            A proxy to a shareable lock.
        heading : str
            A heading for log entries.
        """

        # Process the results
        self.__results[result[1][0]][result[1][1]] = result.success
        _print(log_lock, "{} Saving results from task-{}-{}".format(heading, *result.indices))
        self.__write_object(result)

    def __write_object(self, data):
        """
        Converts an arbitrary picklable object into a sequence of bytes and
        attempts to write it atomically to the database file.

        Parameters
        ----------
        data : object
            Any object that can be pickled in the presence of only the
            paccs package.
        """

        # Prepare a buffer of gzipped pickled data preceded by its own length
        data_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=data_buffer, mode="wb") as archive:
            pickle.dump(data, archive, pickle.HIGHEST_PROTOCOL)
        data_bytes = data_buffer.getvalue()
        data_bytes = _MAGIC_BEGIN + struct.pack("<Q", len(data_bytes)) + data_bytes + _MAGIC_END

        # This save will occur asynchronously, be sure to acquire a lock.
        # io.open with mode "ab" and no buffering should perform a syscall
        # write() with O_APPEND: on POSIX compliant systems, this will
        # hopefully be atomic.
        self.__db_lock.acquire()
        with io.open(self.__db_path, mode="ab", buffering=0) as db_file:
            # Be sure no other TaskManagers are trying to access this file.
            # LOCK_NB is used: this is meant for corruption protection,
            # not to allow multiple TaskManagers to write to the same file.
            try:
                fcntl.lockf(db_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                db_file.write(data_bytes)
                db_file.flush()
                os.fsync(db_file.fileno())
            finally:
                fcntl.lockf(db_file, fcntl.LOCK_UN)
        self.__db_lock.release()

def _work(generate_index, optimize_index, log_lock, preprocessors, postprocessors, optimize_kwargs, group, cell, random_seed):
    """
    A module-level work implementation function to satisfy the requirements
    of the multiprocessing system.
    """

    # Define None values to return in case of an exception
    initial_cell, initial_energy, optimize_result, log_data, start_time, end_time, clock_difference = \
        None, None, None, None, None, None, None
    success = False

    try:
        # Start timing
        start_time, start_clock = time.time(), time.process_time()
        heading = "[task-{}-{}:{}]".format(generate_index, optimize_index, os.getpid()).ljust(20)
        _print(log_lock, "{} Started at {}".format(heading, time.strftime( \
            "%Y-%m-%d %H:%M:%S", time.localtime(start_time))))

        # Prepare for optimization
        initial_cell = cell
        for preprocessor in preprocessors:
            initial_cell = preprocessor(initial_cell)

        optimize_kwargs["cell"] = initial_cell
        optimize_kwargs["random_seed"] = random_seed
        optimize_kwargs["log_file"] = io.StringIO()

        density_preprocessor = [preprocessor for preprocessor in preprocessors
            if type(preprocessor) is CellDensityProcessor]
        if (len(density_preprocessor) == 1):
            optimize_kwargs["enclosed_bounds"] = density_preprocessor[0].enclosed_bounds(optimize_kwargs["cell"])
        elif (len(density_preprocessor) > 1):
            raise ValueError('more than one CellDensityProcessor detected')

        # Calculate the energy per atom of the preprocessed cell
        initial_energy = potential._evaluate_fast(initial_cell, \
            optimize_kwargs["potentials"], optimize_kwargs["distance"])[0]

        # Optimize
        def _postprocess(cell, energy, wall_stamp, processor_stamp):
            postprocessed_cell = cell
            for postprocessor in postprocessors:
                postprocessed_cell = postprocessor(postprocessed_cell)
            if cell is postprocessed_cell:
                return cell, energy, wall_stamp, processor_stamp
            else:
                return postprocessed_cell, potential._evaluate_fast(initial_cell, \
                    optimize_kwargs["potentials"], optimize_kwargs["distance"])[0], \
                    wall_stamp, processor_stamp

        optimize_result = [_postprocess(cell, energy, wall_stamp, processor_stamp) \
            for cell, energy, wall_stamp, processor_stamp in minimization.optimize(**optimize_kwargs)]
        log_data = optimize_kwargs["log_file"].getvalue()

        # End timing
        end_time, end_clock = time.time(), time.process_time()
        clock_difference = end_clock - start_clock
        _print(log_lock, "{} Finished at {}".format(heading, time.strftime( \
            "%Y-%m-%d %H:%M:%S", time.localtime(end_time))))
        _print(log_lock, "{} {} seconds (wall time), {} seconds (process time) elapsed".format( \
            heading, end_time - start_time, clock_difference))

        # Provide the input data as well as the output data for persistence to the database
        success = True
    except Exception:
        # Indicate failure and display a traceback
        success = False
        _print(log_lock, "{} Failure occurred:\n{}".format(heading, traceback.format_exc()))

    return OptimizeResult(success, (generate_index, optimize_index), group, (initial_cell, initial_energy), \
        optimize_result, log_data, start_time, end_time, clock_difference)

class Result:
    pass

class GenerateResult(Result, collections.namedtuple("_GenerateResult", ["index", "log", \
    "accepted_energies", "rejected_energies"])):
    """
    Contains a result produced after exhausting a generator of initial guesses.

    Parameters
    ----------
    index : int
        The index of the generation task.  The first launched
        task has an index of 0.
    log : str
        Logging information produced by the generator.
    accepted_energies : list(float)
        Energies per atom of generated cells accepted by :py:func:`paccs.minimization.filter`.
    rejected_energies : list(float)
        Energies of generated cells rejected by :py:func:`paccs.minimization.filter`.
    """
    pass

class OptimizeResult(Result, collections.namedtuple("_OptimizeResult", ["success", "indices", "symmetry_group", \
    "initial_guess", "candidates", "log", "start_time", "end_time", "processor_time"])):
    """
    Contains a single result produced by a task during a :py:class:`TaskManager` run.

    Parameters
    ----------
    success : bool
        Whether or not the task completed successfully.  If this value is
        False, some fields may be None if the task did not reach the point
        at which they would have been generated.
    indices : tuple(int, int)
        The indices of the generation and optimization task, respectively.
        The first launched generation task has an index of 0.  For each
        generation task, a number of optimization tasks will be created
        with the first having an index of 0.
    symmetry_group : wallpaper.WallpaperGroup
        The symmetry group used to generate the initial guess for the task.
    initial_guess : tuple(paccs.crystal.Cell, float)
        The initial guess provided to the basin-hopping optimizer, along
        with its energy.  The cell has already been preprocessed by this
        point, and the energy corresponds to the preprocessed cell.
    candidates : list(tuple(paccs.crystal.Cell, float, float, float))
        Candidate minima returned from the basin-hopping optimizer, along
        with their energies (per atom), wall timestamps, and processor timestamps.
        The final candidate is the optimization result.  The cell has already
        been postprocessed by this point, and the energy corresponds to the
        postprocessed cell.
    log : str
        Logging information produced by the basin-hopping optimizer.
    start_time : float
        The (Unix epoch) time when the task started executing.
    end_time : float
        The (Unix epoch) time when the task finished executing.
    processor_time : float
        The total processor time in seconds taken by the task.
    """
    pass

class ResultsDatabase:
    """
    Contains tools for manipulating a result database file generated by
    a finished instance of a :py:class:`TaskManager`.

    Parameters
    ----------
    database_path : str
        A path to a database file.  This must be a path; a file object
        is not sufficient.

    Notes
    -----
    Upon creating a :py:class:`ResultsDatabase`, the file will be scanned
    and a basic integrity check will be performed.  Chunks will be loaded
    on demand as they are requested, and then cached for fast retrieval.
    """

    def __init__(self, database_path):
        self.__database_path = database_path

        # Perform initial scan to identify blocks
        self.__block_table = []
        self.__block_cache = {}
        with open(database_path, "rb") as database_file:
            block_length_size = struct.calcsize("<Q")
            while True:
                # Read the starting magic value
                magic_begin = database_file.read(len(_MAGIC_BEGIN))
                if not magic_begin:
                    return # End of file
                if len(magic_begin) != len(_MAGIC_BEGIN):
                    warnings.warn("Database terminated before end of block head", RuntimeWarning)
                elif magic_begin != _MAGIC_BEGIN:
                    warnings.warn("Corruption detected at block head", RuntimeWarning)
                # TODO: seek for next indication of valid block

                # Read the length of the next block
                block_length_bytes = database_file.read(block_length_size)
                if len(block_length_bytes) < block_length_size:
                    warnings.warn("Database terminated before end of block length", RuntimeWarning)
                    return
                block_length = struct.unpack("<Q", block_length_bytes)[0]

                # Skip over the block
                previous_position = database_file.tell()
                database_file.seek(block_length, os.SEEK_CUR)
                next_position = database_file.tell()
                if next_position != previous_position + block_length:
                    warnings.warn("Database terminated before end of block", RuntimeWarning)
                    return

                # Read the ending magic value
                magic_end = database_file.read(len(_MAGIC_END))
                if len(magic_end) != len(_MAGIC_END):
                    warnings.warn("Database terminated before end of block tail", RuntimeWarning)
                    return
                if magic_end != _MAGIC_END:
                    warnings.warn("Corruption detected at block tail", RuntimeWarning)
                # TODO: seek for next indication of valid block

                # Add block information to the list
                self.__block_table.append((previous_position, block_length))

    @property
    def block_count(self):
        """
        Retrieves the number of blocks in the database.

        Returns
        -------
        int
            The number of blocks in the database.  This will typically be
            one greater than the number of structure samples in the database
            (if :py:func:`TaskManager.work` was not interrupted).
        """

        return len(self.__block_table)

    def read_block(self, block_index):
        """
        Retrieves a block with the specified index from the database.

        Parameters
        ----------
        block_index : int
            The zero-based index of the block to retrieve.

        Returns
        -------
        object
            The data stored within.
        """

        if block_index not in self.__block_cache:
            with open(self.__database_path, "rb") as database_file:
                # Read from the file
                database_file.seek(self.__block_table[block_index][0], os.SEEK_SET)
                block_bytes = io.BytesIO(database_file.read(self.__block_table[block_index][1]))

                # Reconstitute the data
                with gzip.GzipFile(fileobj=block_bytes, mode="rb") as archive:
                    self.__block_cache[block_index] = pickle.load(archive)

        return self.__block_cache[block_index]

    @property
    def results(self):
        """
        Retrieves all data in the database.

        Returns
        -------
        generator(Result)
            Results, either :py:class:`GenerateResult` or :py:class:`OptimizeResult`
            objects.
        """

        for block_index in range(self.block_count):
            yield self.read_block(block_index)

    @property
    def generate_results(self):
        """
        Retrieves all results produced by generator runs.

        Returns
        -------
        generator(GenerateResult)
            Results from :py:func:`paccs.wallpaper.generate_wallpaper()`.
        """

        return (result for result in self.results if isinstance(result, GenerateResult))

    @property
    def optimize_results(self):
        """
        Retrieves all results produced by optimizer runs.

        Returns
        -------
        generator(OptimizeResult)
            Results from :py:func:`paccs.minimization.optimize()`.
        """

        return (result for result in self.results if isinstance(result, OptimizeResult))

    @property
    def cells_in(self):
        """
        Retrieves all cells used as initial guesses.

        Returns
        -------
        generator(tuple(paccs.crystal.Cell, float))
            All cells extracted from each of :py:func:`optimize_results`
            that were optimizer inputs, along with their energies (per atom).
        """

        for result in self.optimize_results:
            yield result.initial_guess

    @property
    def cells_out(self):
        """
        Retrieves all cells produced by optimizer runs.

        Returns
        -------
        generator(tuple(paccs.crystal.Cell, float))
            All cells extracted from each of :py:func:`optimize_results`
            that were optimizer outputs, along with their energies (per atom).
        """

        for result in self.optimize_results:
            for candidate in result.candidates:
                # Do not yield discovery timestamps
                yield candidate[:2]
