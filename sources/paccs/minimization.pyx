"""
Applies various algorithms to given crystal cells to attempt to locate local and global energy minima.
"""

from . import crystal
from . import potential
from . import wallpaper
from . import similarity
import multiprocessing
import math
import numpy
import scipy
import scipy.optimize
import sklearn.cluster
import sys
import time
import warnings

def _filter_energy(index, cell, radii, potentials, distance):
    scale_cell = lambda cell: cell if radii is None else \
        crystal.CellTools.scale(cell, cell.vectors / cell.scale_factor(radii))
    return (index, potential._evaluate_fast(scale_cell(cell), potentials, distance)[0])

def filter(generator, potentials, distance, count, radii=None,
    histogram_callback=None,
    similarity_metric=None,
    pool=None):
    r"""
    Filters a generator produced by :py:func:`paccs.wallpaper.generate_wallpaper` by
    calculating energies, and potentially other metrics, of cells and only permitting a certain
    (structurally unique) fraction with sufficiently low energies to pass.

    Parameters
    ----------
    generator : generator(tuple(paccs.wallpaper.WallpaperGroup, paccs.crystal.Cell))
        A generator from :py:func:`paccs.wallpaper.generate_wallpaper`.  Any generator
        that provides a similar input can also be used.
    potentials : dict(tuple(int or str), paccs.potential.Potential)
        For energy evaluation.  See :py:func:`paccs.crystal.CellTools.energy`.
    distance : float
        For energy evaluation.  See :py:func:`paccs.crystal.CellTools.energy`.
    count : int
        The number of cells to allow through.  The cells with the lowest energies
        will be yielded.
    radii : tuple(float)
        The radii of atom types.  If specified, performs automatic rescaling to contact
        before energy evaluation.
    histogram_callback : callable
        If it is desired to examine the energies of the cells produced after
        the filtering operation is complete, provide a callable which will
        be invoked at exhaustion of this generator.  It should accept a list
        of all energies.  The first **count** elements will be the lowest
        in energy and will have been yielded in that order as well.
        No other guarantees as to the order of the energies are made.
    similarity_metric : :py:class:`~paccs.similarity.SimilarityMetric`
        Similarity metric to use to remove structurally redundant configurations.
        Recall that :py:class:`~paccs.similarity.Hybrid` can be used to chain several
        metrics together.  Similarity is evaluated **after** scaling if radii are
        specified.
    pool : multiprocessing.Pool
        Pool to parallelize energy calculation, if desired.

    Returns
    -------
    generator(tuple(paccs.wallpaper.WallpaperGroup, paccs.crystal.Cell))
        A generator yielding cells in the same manner as
        :py:func:`paccs.wallpaper.generate_wallpaper`, only filtered.

    Notes
    -----
    **This will fully exhaust the generator** that is provided so that it may not be
    used any longer.  Furthermore, if radii is/are specified and rescaling is performed,
    the energy and similarity metrics (if specified) are computed based on the rescaled
    cells, however, the cells that will be provided by the resulting generator are the
    same as the **original ones** produced by the input generator. In other words, scaling
    must be done again on the those cells provided in the resulting generator if desired.
    Thus, filtering has no effect of changing the output of the generator, only
    removal of certain entries, essentially.
    """

    # Automatically scales a cell before calculation if desired
    scale_cell = lambda cell: cell if radii is None else \
        crystal.CellTools.scale(cell, cell.vectors / cell.scale_factor(radii))

    # Calculate all energies to determine the cutoff exactly
    results = list(generator)

    if (pool is None):
        energies = numpy.array([potential._evaluate_fast(scale_cell(cell), potentials, distance)[0] \
            for group, cell in results])
    else:
        energies = numpy.empty(len(results), dtype=numpy.float64)
        def _callback(x):
            energies[x[0]] = x[1]
        multi_ener = [pool.apply_async(_filter_energy, (idx, cell, radii, potentials, distance), callback=_callback) for idx, (group, cell) in enumerate(results)]

        # Block until these energy calculations have finished
        for me in multi_ener:
            try:
                me.get()
            except Exception as e:
                raise Exception("error during filtering process: {}".format(e))

    # Apply similarity if desired
    if (similarity_metric is not None):
        # This is a  bit hackish since SimilarityMetrics expect tuple of (cell, energy)
        # but only use values 0 or 1 of this tuple, we can send "ghost" information
        # as appended values to this tuple, such as the WallpaperGroup. This way,
        # that information is always kept together as the reduction is performed.
        # In fact, documentation on similarity.reduce() stipulates that the metric
        # should expect to work with only 2 tuple entries, so technically this is
        # sound.
        reduced_triplet = similarity.reduce([(scale_cell(results[i][1]), energies[i], results[i][0], i) for i in range(len(results))], similarity_metric) # Sorts by energy and metric
        sorted_data = [((x[2], results[x[3]][1]), x[1]) for x in reduced_triplet]
    else:
        # Otherwise, just sort by energy
        sorted_data = sorted(zip(results, energies), key=lambda pair: pair[1])

    # Filter and yield the cells
    for index, (result, energy) in enumerate(sorted_data):
        if index == count:
            break
        yield result

    # Invoke the callback if one is present
    if histogram_callback is not None:
        histogram_callback([energy for result, energy in sorted_data])

def _decompose_cell(cell):
    """
    Converts a cell object into a flat vector representation.

    Parameters
    ----------
    cell : paccs.crystal.Cell
        The cell to decompose.  It is expected that this cell will be
        fully reduced (see :py:func:`paccs.crystal.CellTools.reduce`).  If
        this is not the case, energy evaluation may take an excessive
        amount of time and generate incorrect results or optimization
        may fail to converge.

    Returns
    -------
    numpy.ndarray
        A one-dimensional array containing information about the cell
        vectors and atom coordinates.
    """

    # Important information on the structure of this vector: it consists in sequence of
    # [ lower triangular elements of cell row vector matrix ] +
    # [ atom type 0 pos 1 (x y z) ... atom type 0 pos N (x y z) ] +
    # [ atom type 1 pos 0 (x y z) ... atom type 1 pos N (x y z) ] +
    # [ atom type N pos 0 (x y z) ... atom type N pos N (x y z) ]
    # Note specifically that atom type 0 pos 0 is absent: if the cell is fully
    # reduced as is required to call _decompose_cell, then this is always at (0 0 0)

    atom_lists = cell.atom_lists
    atom_lists[0] = atom_lists[0][1:]
    return numpy.concatenate([cell.vectors[numpy.tril_indices(cell.dimensions)], \
        numpy.concatenate(atom_lists).flatten()])

def _recompose_cell(template_cell, vector):
    """
    Converts a flat vector representation of a cell object back into
    an object.

    Parameters
    ----------
    template_cell : paccs.crystal.Cell
        A template cell from which the number of dimensions and
        atoms of each type are drawn to reconstitute the cell.  It is
        expected that this cell will be fully reduced (see
        :py:func:`paccs.crystal.CellTools.reduce`).  If this is not the
        case, energy evaluation may take an excessive amount of time
        and generate incorrect results or optimization may fail to converge.
    vector : ndarray
        A one-dimensional array generated by :py:func:`_decompose_cell`.

    Returns
    -------
    paccs.crystal.Cell
        The reconstituted cell.
    """

    # Extract cell vectors
    vectors = numpy.zeros((template_cell.dimensions, template_cell.dimensions))
    indices = numpy.tril_indices(template_cell.dimensions)
    index = indices[0].shape[0]
    vectors[indices] = vector[:index]

    # Extract atom lists
    atom_lists = []
    for type_index in range(template_cell.atom_types):
        if type_index == 0:
            # Special case: the first atom is at the origin and is not explicitly stored here
            next_index = index + ((template_cell.atom_count(type_index) - 1) * template_cell.dimensions)
            atom_lists.append(numpy.concatenate([numpy.zeros((1, template_cell.dimensions)), vector[index:next_index] \
                .reshape(template_cell.atom_count(type_index) - 1, template_cell.dimensions)]))
            index = next_index
        else:
            # Normal case
            next_index = index + (template_cell.atom_count(type_index) * template_cell.dimensions)
            atom_lists.append(vector[index:next_index] \
                .reshape(template_cell.atom_count(type_index), template_cell.dimensions))
            index = next_index

    # Wrap atoms so that energy evaluation does not produce invalid results
    # Optimize cell geometry while being careful to avoid changes that invalidate Jacobian
    return crystal.CellTools.reduce(crystal.Cell(vectors, atom_lists, template_cell.names), \
        normalize=False, condense=False)

def optimize(cell, potentials, distance, log_level=0, log_file=sys.stderr, log_grad=False, \
    enclosed_bounds=None, surface_bounds=None, distortion_factor_bounds=(1.0, 10.0), \
    vector_delta=2.0 ** -26.0, exchange_move=0.2, exchange_select=0.5, vector_move=0.2, \
    vector_select=0.5, vector_shear=0.5, vector_factor=1.5, scale_move=0.1, scale_factor=1.5, \
    atom_move=0.4, atom_select=0.5, cluster_move=0.1, cluster_factor=0.5, initial_step=0.1, \
    random_seed=None, initial_kwargs=dict(), basin_kwargs=dict(), final_kwargs=dict(), \
    save_count=1, save_all=False, save_filter=None, retain_timestamp=False, _DEBUG_XYZ_PATH=None):
    """
    Uses basin-hopping to  perform structure optimization given pairwise
    potentials and an initial structure as a guess.

    Parameters
    ----------
    cell : paccs.crystal.Cell
        The cell used as an initial guess for the optimization.
    potentials : dict(tuple(int or str), paccs.potential.Potential)
        Potentials for each pair of atom types.  If a potential is not specified
        for a given pair, then no interactions will occur between those atoms.
    distance : float
        Distance to which potential energy evaluation should take place (global r_cut).
    log_level : int
        Verbosity of diagnostic messages to display.  By default, nothing is displayed.
        Increasing this value leads to more verbose output; setting it to -1 displays
        all messages.
    log_file : file
        Where to send diagnostic messages.  By default, they are sent to the standard
        error stream.
    log_grad : bool
        Approximate the gradient of the objective function using finite differences
        and display the error in the analytically computed gradient on each optimization
        step.  Note that no output will be produced if the `log_level` is too low.
    enclosed_bounds : tuple(float)
        Lower and upper bounds on the space (area or volume) enclosed by the cell during a local
        relaxation step.
    surface_bounds : tuple(float)
        Lower and upper bounds on the surface space of the cell during a local
        relaxation step.
    distortion_factor_bounds : tuple(float)
        Lower and upper bounds on the distortion factor of the cell during a
        local relaxation step.
    vector_delta : float
        Amount by which to adjust cell vector components when estimating the gradient
        during a local relaxation step.
    exchange_move : float
        Relative probability of an atom exchange move during basin-hopping.
    exchange_select: float
        The approximate probability that an atom will be selected during an exchange
        move (binomial probability).  To avoid complex calculations in selecting swappable pairs of atoms,
        the actual probability may deviate slightly. If this is either zero or negative,
        only a single pair of (different) atoms is chosen to be swapped.  All swaps only
        occur between different types of atoms, since swapping two of the same would
        have not net effect.
    vector_move : float
        Relative probability of a vector perturbation move during basin-hopping.
    vector_select : float
        Probability that a given vector component will be selected for modification
        if a vector perturbation move is chosen.
    vector_shear : float
        Probability that atoms will be moved (affine deformation) after vector modification, else is a non-affine deformation.
    vector_factor : float
        Maximum adjustment factor for vector moves.  Should be greater than 1.
        When set (for example) to 1.4, expect to see up to approximately 40%
        increases and decreases in vector components on a vector move.
    scale_move : float
        Relative probability of a uniform scaling (expansion or contraction) move.
    scale_factor : float
        Maximum adjustment factor for uniform scaling moves.  Should be greater
        than 1.  When set (for example) to 1.4, expect to see up to approximately
        40% increases and decreases in scale on a scaling move.
    atom_move : float
        Relative probability of a atom perturbation move during basin-hopping.
    atom_select : float
        Probability that a given atom coordinate will be selected for motion if an atom
        perturbation move is chosen.
    cluster_move : float
        Relative probability of a clustered move during basin-hopping.
    cluster_factor : float
        Maximum adjustment factor for cluster moves.  When set (for example) to
        0.4, expect to see cluster moves in a given direction of up to 40%
        of the distance between cluster centroids.
    initial_step : float
        Initial step size for perturbation moves.  This size may be adjusted by
        the basin-hopping algorithm during operation via ``basin_kwargs`` interval parameter.
        For numerical stability, this is also forcibly bounded between 0 < initial_step < max([initial_step*100, numpy.finfo.max/2.0]).
        If initial_step is negative, the absolute value will be used instead.
    random_seed : int
        The seed for the random number generator (which uses the Mersenne Twister algorithm).
        If not specified, a random seed will be chosen.  Specify this value if it is desired to
        generate exactly reproducible results.  It should be non-negative and less than :math:`2^{32}`.
    initial_kwargs : dict(str, object)
        Keyword arguments to provide to the optimizer for the initial relaxation
        stage of the optimization.  These are documented; see :py:func:`scipy.optimize.minimize`.
        By default, the L-BFGS-B algorithm is used.
    basin_kwargs : dict(str, object)
        Keyword arguments to provide to the optimizer for the basin-hopping stage
        of the optimization.  These are documented; see :py:func:`scipy.optimize.basinhopping`.
        By default, the L-BFGS-B algorithm is used for relaxation.
    final_kwargs : dict(str, object)
        Keyword arguments to provide to the optimizer for the final relaxation stage
        of the optimization.  This may be useful if it is desired to minimize using a large
        tolerance during basin-hopping and then refine the structure with a small tolerance
        at the very end.  By default, the Nelder-Mead algorithm is used.
    save_count : int or None
        The maximum number of structures to save.  If this value is greater than 1, structures
        generated during basin-hopping will be returned in addition to the final structure.
        If this value is set to None, all structures corresponding to accepted moves will be
        returned.
    save_all : bool
        When set to True, every relaxed structure obtained will be considered a candidate for
        saving (instead of only those satisfying the basin-hopping acceptance criterion).
    save_filter : paccs.similarity.SimilarityMetric
        If specified, only dissimilar cells as indicated by the provided similarity metric
        will be saved.  Otherwise, all cells will be saved.  Cells with the lowest energies
        will be retained.
    retain_timestamp : bool
        With a value of False, if a new structure is discovered with lower energy but
        similar to an existing structure, the timestamp of the old structure will be
        discarded when the new one is saved.  Setting this parameter to True keeps the
        new structure with the existing timestamp.

    Returns
    -------
    list(tuple(paccs.crystal.Cell, float, float, float))
        Optimized structures, their energies (per atom), the wall times at which they were
        discovered, and the processor times at which they were discovered.  The last
        structure will always be the final structure after the last relaxation.  There
        may be other structures present; if so, their energies will be provided as well.
    """

    # Undocumented parameters
    # _DEBUG_XYZ_PATH : str
    #     When specified, dumps out the current cell to an XYZ file on every
    #     objective function evaluation.  This can make extremely large files.

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
    log(1, "@Preparing for optimization@")

    # Process bounds
    if enclosed_bounds is None: enclosed_bounds = (0.0, numpy.inf)
    if surface_bounds is None: surface_bounds = (0.0, numpy.inf)
    if distortion_factor_bounds is None: distortion_factor_bounds = (0.0, numpy.inf)

    # Check probabilities
    type_lookup = [type_index \
        for type_index in range(cell.atom_types) \
        for atom_index in range(cell.atom_count(type_index))]
    total_atoms = len(type_lookup)
    if total_atoms == 1 and atom_move:
        warnings.warn("single atom present; no atom moves will be made", RuntimeWarning)
        atom_move = 0
    if total_atoms == 1 and cluster_move:
        warnings.warn("single atom present; no cluster moves will be made", RuntimeWarning)
        cluster_move = 0
    if cell.atom_types == 1 and exchange_move:
        warnings.warn("single atom type present; no exchange moves will be made", RuntimeWarning)
        exchange_move = 0

    initial_step = numpy.abs(initial_step)

    # Normalize probabilities
    probability_sum = exchange_move + vector_move + atom_move + scale_move + cluster_move
    exchange_move /= probability_sum
    vector_move /= probability_sum
    scale_move /= probability_sum
    atom_move /= probability_sum
    cluster_move /= probability_sum

    # Initialize random source
    randomness = numpy.random.RandomState(random_seed)
    choices = numpy.array([exchange_move, vector_move, scale_move, atom_move, cluster_move])

    # Prepare indices for referencing into the decomposed vector
    # See _decompose_cell for information about vector structure
    total_dimensions = cell.dimensions
    total_components = total_dimensions * (total_dimensions + 1) // 2

    # Check minimizer options and set defaults
    def accepts_jacobian(method):
        if method in {"dogleg", "trust-ncg"}:
            raise ValueError("analytical Hessian calculation not supported")
        return method in {"CG", "BFGS", "Newton-CG", "L-BFGS-B", "TNC", "SLSQP"}

    initial_kwargs = dict(initial_kwargs)
    if "method" not in initial_kwargs:
        initial_kwargs["method"] = "L-BFGS-B"
    initial_kwargs["jac"] = initial_jac = accepts_jacobian(initial_kwargs["method"])

    basin_kwargs = dict(basin_kwargs)
    if "minimizer_kwargs" not in basin_kwargs:
        basin_kwargs["minimizer_kwargs"] = dict()
    if "method" not in basin_kwargs["minimizer_kwargs"]:
        basin_kwargs["minimizer_kwargs"]["method"] = "L-BFGS-B"
    basin_kwargs["minimizer_kwargs"]["jac"] = basin_jac = accepts_jacobian(basin_kwargs["minimizer_kwargs"]["method"])

    final_kwargs = dict(final_kwargs)
    if "method" not in final_kwargs:
        final_kwargs["method"] = "Nelder-Mead"
    final_kwargs["jac"] = final_jac = accepts_jacobian(final_kwargs["method"])

    # Define the optimization parameters
    return_jac = True
    def objective_function(vector):
        # Make a cell object from the current vector
        recomposed = _recompose_cell(cell, vector)
        enclosed = recomposed.enclosed
        surface = recomposed.surface
        distortion_factor = recomposed.distortion_factor
        log(3, "@Objective function evaluation with V={}, A={}, f={}".format(enclosed, surface, distortion_factor))

        # Dump output to an XYZ if necessary for debugging
        if _DEBUG_XYZ_PATH:
            with open(_DEBUG_XYZ_PATH, "a") as debug_xyz_file:
                crystal.CellCodecs.write_xyz(recomposed, debug_xyz_file)

        # Make sure the optimizer didn't make an absurd move
        if not enclosed_bounds[0] <= enclosed <= enclosed_bounds[1] \
            or not surface_bounds[0] <= surface <= surface_bounds[1] \
            or not distortion_factor_bounds[0] <= distortion_factor <= distortion_factor_bounds[1]:
            log(3, ": rejecting bad attempt by local optimizer@")
            return (numpy.inf, numpy.zeros_like(vector)) if return_jac else numpy.inf

        # Do the evaluation
        energy, partial_jacobian = potential._evaluate_fast(recomposed, potentials, distance)
        log(3, ": got E/N={}".format(energy))
        if not return_jac:
            log(3, "@")
            return energy
        partial_jacobian = partial_jacobian[total_dimensions:]

        # Perturb each cell vector slightly to estimate the Jacobian
        vector_jacobian = numpy.zeros(total_components)
        for index in range(vector_jacobian.shape[0]):
            new_vector = numpy.array(vector)
            new_vector[index] += vector_delta
            new_energy = potential._evaluate_fast(_recompose_cell(cell, new_vector), potentials, distance)[0]
            vector_jacobian[index] = (new_energy - energy) / vector_delta
        total_jacobian = numpy.concatenate([vector_jacobian, partial_jacobian])

        components = numpy.abs(total_jacobian)
        log(3, ", |d(E/N)|={}, max|d(E/N)i|={}, min|d(E/N)i|={}@".format( \
            numpy.linalg.norm(total_jacobian), max(components), min(components)))
        if log_grad:
            approx_jacobian = scipy.optimize.approx_fprime(vector, \
                lambda vector: potential._evaluate_fast(_recompose_cell(cell, vector), \
                potentials, distance)[0], vector_delta)
            log(4, "@Gradient error approximation={}%@".format(100 * \
                numpy.linalg.norm(approx_jacobian - total_jacobian) / (numpy.linalg.norm(approx_jacobian))))
        return energy, total_jacobian

    # Define a custom step taking class
    step_count = 1
    class StepTaker:
        def __init__(self, stepsize=initial_step):
            self.stepsize = stepsize
            self.__stepsize_bound = numpy.min([initial_step*100, numpy.finfo(float).max/2.0])
        def __call__(self, vector):
            nonlocal step_count
            log(2, "@Basin-hopping step {} with dx={}".format(step_count, self.stepsize))
            step_type = randomness.choice(len(choices), p=choices)

            if (self.stepsize > self.__stepsize_bound):
                log(2, "@Basing-hopping step size reached upper bound, reset from {} to {}".format(self.stepsize, self.__stepsize_bound))
                self.stepsize = self.__stepsize_bound
            elif (self.stepsize < numpy.finfo(float).eps):
                log(2, "@Basing-hopping step size reached lower bound, reset from {} to {}".format(self.stepsize, initial_step))
                self.stepsize = initial_step

            if step_type == 0: # exchange_move (see definition of choices array)

                if (exchange_select <= 0):
                    # Perform a single pairwise exchange
                    exchange_count = 1
                else:
                    # Perform a number of exchanges (there will be a non-zero probability of swapbacks)
                    exchange_count = 0
                    while exchange_select and not exchange_count:
                        exchange_count = int(numpy.round(randomness.binomial(total_atoms, exchange_select) / 2))
                log(2, ": exchanging atoms")

                for exchange_index in range(exchange_count):
                    # Choose two different atoms to exchange
                    index_1, index_2 = 0, 0
                    while type_lookup[index_1] == type_lookup[index_2]:
                        index_1, index_2 = randomness.randint(total_atoms, size=2)
                    if index_1 > index_2:
                        index_1, index_2 = index_2, index_1
                    log(2, "{} {} ({}) and {} ({})".format("," if exchange_index else "", \
                        index_1, cell.name(type_lookup[index_1]), index_2, cell.name(type_lookup[index_2])))

                    if index_1 == 0:
                        # The zero atom is not stored directly in the vector; shift all atoms as necessary
                        # to simulate moving the zero atom and then do a "swap" with the other atom
                        index_2 = ((index_2 - 1) * total_dimensions) + total_components
                        vector_shift = vector[index_2:index_2 + total_dimensions].copy()
                        vector[total_components:] -= numpy.tile(vector_shift, total_atoms - 1)
                        vector[index_2:index_2 + total_dimensions] = -vector_shift
                    else:
                        # Swap coordinates of the selected atoms
                        index_1 = ((index_1 - 1) * total_dimensions) + total_components
                        index_2 = ((index_2 - 1) * total_dimensions) + total_components
                        vector[index_1:index_1 + total_dimensions], vector[index_2:index_2 + total_dimensions] = \
                            vector[index_2:index_2 + total_dimensions], vector[index_1:index_1 + total_dimensions].copy() #!
                log(2, "@")

            elif step_type == 1: # vector_move

                # Determine the probability of selection for this particular move
                vector_probability = randomness.uniform(numpy.finfo(float).eps, vector_select)

                # Choose some random components to modify
                components_moving = numpy.array([])
                while vector_select and not components_moving.shape[0]:
                    components_moving = numpy.arange(total_components)[ \
                        randomness.choice([False, True], size=total_components, p=[1 - vector_probability, vector_probability])]
                log(2, ": moving {} vector component{}@".format(components_moving.shape[0], "" if components_moving.shape[0] == 1 else "s"))

                # Determine whether or not a shear move should be made
                do_vector_shear = randomness.uniform() < vector_shear

                # Save the old components if they are needed to move the atoms with the vectors
                if do_vector_shear:
                    old_components = vector[:total_components].copy()

                # Make the move itself (step size is based on the largest vector component)
                vector_step_base = numpy.max(numpy.abs(vector[:total_components]))
                vector_step_plus = vector_step_base * (vector_factor - 1)
                vector_step_minus = -vector_step_plus 
                vector[components_moving] += randomness.uniform(vector_step_minus, vector_step_plus, components_moving.shape[0])

                # Move the atoms along with the vectors if necessary
                if do_vector_shear:
                    # Do not go through crystal.CellTools.scale (_decompose assumes it will be getting a
                    # lower triangular cell vector matrix and will output invalid data otherwise; this will
                    # not be the case for the output of scale() in general).  Instead, do the transformation
                    # directly on the decomposed vectors.

                    # Convert lower triangular element arrays to full matrices
                    old_matrix = numpy.zeros((cell.dimensions, cell.dimensions))
                    old_matrix[numpy.tril_indices(cell.dimensions)] = old_components
                    new_matrix = numpy.zeros((cell.dimensions, cell.dimensions))
                    new_matrix[numpy.tril_indices(cell.dimensions)] = vector[:total_components]

                    # Perform the transformation itself
                    old_atoms = vector[total_components:].reshape((-1, cell.dimensions))
                    new_atoms = numpy.dot(new_matrix.T, numpy.dot(scipy.linalg.inv(old_matrix.T), old_atoms.T)).T
                    vector[total_components:] = new_atoms.flatten()

            elif step_type == 2: # scale_move

                # Scale the entire vector: this has the cell vectors and the coordinates within
                log(2, ": performing uniform scale@")
                vector *= scale_factor ** randomness.uniform(-1.0, 1.0)

            elif step_type == 3: # atom_move

                # Determine the probability of selection for this particular move
                atom_probability = randomness.uniform(numpy.finfo(float).eps, atom_select)

                # Choose some random coordinates to modify
                coordinates_choice = [False]
                while atom_select and not any(coordinates_choice):
                    coordinates_choice = randomness.choice([False, True], size=total_atoms * total_dimensions, p=[1 - atom_probability, atom_probability])
                coordinates_moving = numpy.arange(vector.shape[0] - total_components)[coordinates_choice[total_dimensions:]] + total_components
                moving_count = coordinates_moving.shape[0] + numpy.sum(coordinates_choice[:total_dimensions])
                log(2, ": moving {} atom coordinate{}@".format(moving_count, "" if moving_count == 1 else "s"))

                # Make the move for the first atom (by moving all other atoms by the same amount)
                vector[total_components:] -= numpy.tile(coordinates_choice[:total_dimensions] * randomness.uniform( \
                    -self.stepsize, self.stepsize, total_dimensions), total_atoms - 1)

                # Make the move for atoms other than the first
                vector[coordinates_moving] += randomness.uniform(-self.stepsize, self.stepsize, coordinates_moving.shape[0])

            elif step_type == 4: # cluster_move

                # Use k-means clustering (probably very bad for actual cluster identification
                # but should grab multiple groups of atoms to move in a coordinated fashion)
                log(2, ": performing cluster move@")
                coordinates = numpy.concatenate([numpy.zeros((1, total_dimensions)), numpy.reshape(vector[total_components:], (total_atoms - 1, total_dimensions))])
                k_means = sklearn.cluster.KMeans(n_clusters=2, random_state=randomness, copy_x=False).fit(coordinates)

                # Determine the maximum move amount and make the move in the coordinate array
                cluster_step_max = cluster_factor * numpy.linalg.norm(k_means.cluster_centers_[1] - k_means.cluster_centers_[0])
                atom_deltas = randomness.uniform(-cluster_step_max, cluster_step_max, (2, total_dimensions))[k_means.labels_]

                # Make the move for the first atom, then for all other atoms
                vector[total_components:] -= numpy.tile(atom_deltas[0], total_atoms - 1)
                vector[total_components:] += atom_deltas[1:].flatten()

            else:
                # This should never happen
                raise RuntimeError("internal failure: unsupported step type")

            try:
                vector = _decompose_cell(crystal.CellTools.reduce(_recompose_cell(cell, vector), condense=False))
            except Exception as e:
                raise Exception("unable to complete move because {}".format(e))

            step_count += 1
            return vector

    # Prepare callback for basin-hopping
    basin_results = []
    def basin_callback(vector, energy, accepted, force=False):
        returned_cell = crystal.CellTools.reduce(_recompose_cell(cell, vector), condense=False)

        if not force: # Callback from basin-hopping (we may or may not add the cell)
            # NaN comparisons will all return False and can result in bogus structures
            # getting inserted into the list
            if not numpy.isfinite(energy):
                return

            # Save accepted moves unless explicitly overridden
            if not (accepted or save_all):
                return

        # Add the new cell and force it to the front of the list if desired
        basin_results.append((returned_cell, energy, time.time() - start_time, time.process_time() - start_clock))
        sort_key = lambda sort_cell: -numpy.inf if force and sort_cell is returned_cell else sort_cell[1]

        # Apply the algorithm used by similarity.reduce
        sorted_results = sorted(basin_results, key=sort_key)
        del basin_results[:]
        for test_cell in sorted_results:
            for result_index, result_cell in enumerate(basin_results):
                # If a match is found, a lower energy cell similar to this one
                # is already present in the list of results
                if save_filter is not None and save_filter(test_cell[:2], result_cell[:2]):
                    # If the cell already in the list is the new one, and we want to
                    # preserve the old discovery timestamp, make the change to that cell
                    if retain_timestamp and result_cell is returned_cell:
                        basin_results[result_index] = result_cell[:2] + test_cell[2:]
                    # Do not add this cell to the list
                    break
            else:
                # No match was found; add this cell back into the list
                basin_results.append(test_cell)

        # Trim the results list if it has grown too large
        # The highest energy cell should be at the very end
        if save_count is not None:
            del basin_results[save_count:]

    # Add unrelaxed cell to list in case optimization is unstable and fails
    starting_point = _decompose_cell(cell)
    return_jac = False
    basin_callback(starting_point, objective_function(starting_point), True, force=True)

    # Perform initial relaxation
    log(1, "@Performing initial relaxation@")
    cell = crystal.CellTools.reduce(cell, condense=False)
    return_jac = initial_jac
    initial_guess = scipy.optimize.minimize(objective_function, _decompose_cell(cell), **initial_kwargs).x

    try:
        # If niter is specified in basin hopping, check if it is 0; if so, skip that step altogether
        do_basin_hopping = (basin_kwargs['niter'] > 0)
    except KeyError:
        # Default to try basin hopping
        do_basin_hopping = True

    if do_basin_hopping:
        # Do basin-hopping
        log(1, "@Performing basin-hopping@")
        cell = crystal.CellTools.reduce(_recompose_cell(cell, initial_guess), condense=False)
        return_jac = basin_jac
        refined_guess = scipy.optimize.basinhopping(objective_function, initial_guess, take_step=StepTaker(), callback=basin_callback, **basin_kwargs).lowest_optimization_result.x
    else:
        refined_guess = initial_guess

    # Perform final relaxation
    log(1, "@Performing final relaxation@")
    cell = crystal.CellTools.reduce(_recompose_cell(cell, refined_guess), condense=False)
    return_jac = final_jac
    final_guess = scipy.optimize.minimize(objective_function, _decompose_cell(cell), **final_kwargs).x

    # Get final structure
    log(1, "@Returning final optimized structure@")
    return_jac = False
    basin_callback(final_guess, objective_function(final_guess), True, force=False)

    # In the future, a screen here could be added to remove any NaN or inf energy structures.
    # At the moment, it is better that this is returned so that the user is made aware this
    # behavior is occurring.

    # Reverse list so that lowest energy structure is at the end
    end_time, end_clock = time.time(), time.process_time()
    log(1, "@Optimization routine wall time t={}s@".format(end_time - start_time))
    log(1, "@Optimization routine processor time t={}s@".format(end_clock - start_clock))
    return list(reversed(basin_results))
