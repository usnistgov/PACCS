# This file is available for use in /docs/source/_static/run.py

# Add the paccs modules to the Python path.  You should update this
# in accordance with your paccs installation, or remove it completely if
# you know that paccs will be available on your $PYTHONPATH.
import sys
sys.path.append("../../../lib")

# Import paccs.  If desired, you can also import only the components
# which you plan on using, or use `import paccs` to keep the package
# more self-contained.
from paccs import *

task_manager = automation.TaskManager(
    # Each element here should be a tuple(dict, dict) representing a job
    # to work on.  The first dict contains arguments for generate_wallpaper(),
    # and the second dict contains arguments for optimize().
    [(dict(
        # Arguments for paccs.wallpaper.generate_wallpaper().

        # Required argument specifying the stoichiometry for the problem.
        stoichiometry=(1, 2),

        # Store run information in the output database for later viewing.
        log_level=2,

        # Limits the number of particles placed in a wallpaper group's
        # fundamental domain, or tile.
        place_max=5,

        # The resolution of the discretized grids used on the fundamental domain.
        grid_count=5,

        # The number of cells to generate.
        sample_count=100,

        # Make fundamental domains of cells with a node density "congruent",
        # or similar to, a p1 cell with the above grid_count value.
        congruent=False,

        # If desired, you can specify the wallpaper groups to use.  If this
        # parameter is not present, all seventeen groups will be included.
        # , sample_groups=[
        #     wallpaper.WallpaperGroup(name="p3"),
        #     wallpaper.WallpaperGroup(name="p4")]
    ), dict(
        # Arguments for paccs.minimization.optimize().

        # The pair potentials to use, and the energy evaluation cutoff.
        potentials={
            (0, 0): potential.LennardJonesType(lambda_=-0.25),
            (1, 1): potential.LennardJonesType(lambda_=0.25),
            (0, 1): potential.LennardJonesType(lambda_=1.0)},
        distance=6.0,
        # These are NOT "LJ-lambda" potentials!  Consult the documentation for
        # paccs.potential.LennardJonesType to view the exact form of the
        # potential.  To construct an LJ-lambda potential, you should combine
        # two instances of paccs.potential.LennardJonesType using
        # paccs.potential.Piecewise.

        # Store run information in the output database for later viewing.
        log_level=2,

        # Initial step size for particle displacement moves (this parameter
        # is automatically adjusted by the basin-hopping algorithm).
        initial_step=0.4,

        # Options for the basin-hopping optimizer.
        basin_kwargs=dict(
            # The temperature to use in the Metropolis criterion.
            T=0.005,

            # The interval (in iterations) to adjust the step size.
            interval=10,

            # The number of iterations to terminate after if the best
            # candidate for the minimum has not changed.
            niter_success=100,

            # The number of iterations to terminate after unconditionally.
            niter=10000,

            # A significant increase in speed may be had if the minimization
            # after each perturbation is **not** required to be as strict as the
            # default settings.  Note that the initial and final (best) guess
            # will be refined with different settings as given by initial_kwargs
            # and final_kwargs in paccs.minimization.optimize().
            minimizer_kwargs={"method":"L-BFGS-B", "options":{"gtol":3.0e-2,
            "ftol":3.0e-2, "maxiter":100}}
        ),

        # How many structures in total to retain (including the lowest energy
        # structure).
        save_count=10,

        # Whether or not to save structures resulting from rejected hops.
        save_all=True,

        # Bounds (min,max) for the area of the (super)cell during optimization.
        # Note that these values bound the supercell area so that if a
        # TilingProcessor and/or ScalingProcessor are used one should ensure
        # that these bounds are adequate for the NxM cell, not just a 1x1 cell.
        # A CellDensityProcessor will override these bounds, if one is specified.
        enclosed_bounds=(0.0, 100.0),

        # A filter used to prevent duplicate structures from being saved.
        save_filter=similarity.Histogram(
            distance=6.0,
            bin_width=0.2,
            threshold=0.95,
            norm=similarity.Minimum())

        # If desired, you can output the optimization trajectory for debugging.
        # For even a small production-type run such as the one described in this
        # file, this is a very bad idea as the file will be incredibly large.
        # , _DEBUG_XYZ_PATH="optimization_trajectory.xyz"
    ))],

    # These preprocessors get executed on cells before optimization.
    preprocessors = [
        # Calls crystal.CellTools.scale and crystal.Cell.scale_factor
        # to perform a rescaling so that generated cells have contacts.
        automation.ScalingProcessor(radii=(0.5, 0.5)),

        # A special preprocessor which breaks up jobs per wallpaper group,
        # and causes paccs.minimization.filter() to be invoked to
        # save only a certain number of generated cells.
        automation.AutoFilteringProcessor(count=10),

        # Calls crystal.CellTools.tile to create a supercell.
        automation.TilingProcessor((2, 2))

        # Can impose bounds on the area allowable for the cell while being
        # optimized based on a desired range of densities. This will override
        # any enclosed_bounds() specified above, though. These bounds are applied
        # on the scaled, tiled cell if a ScalingProcessor and/or TilingProcessor
        # is/are used.
        #, automation.CellDensityProcessor((0.1, numpy.inf))
    ],

    # These postprocessors get executed on cells after optimization.
    postprocessors = [
        # All processors including custom subclasses of automation.CellProcessor
        # (except for automation.AutoFilteringProcessor) can be used here as well.
    ],

    # Specifying a random integer in [0, 2^32 - 1] for reproducible results.
    # If this is left out, a new seed will be used on every run.
    random_seed = 123456789,

    # If desired, you can specify a fixed number of worker processes.  By
    # default, paccs automatically detects the number of cores available
    # and uses that value.
    # , worker_count = 2
)

# Perform the run itself
task_manager.work("output.db")
