"""
Tools to analyze or generate ensembles of structures.
"""

import os
import sys
import multiprocessing
import numpy as np
from multiprocessing import Pool
from . import minimization
from . import wallpaper

# Recommended length ratios for sampling.  Where length ratios are fixed, 1 is
# specified, however, it will be ignored anyway.  This is expected to be the ratio
# of the longer edge over the shorter one, such that the ratio is always >= 1.
RECOMMENDED_LENGTH_RATIOS = {
    'p1': (1.0, np.sqrt(2), np.sqrt(3), 2.0),
    'p2': (1.0, np.sqrt(2), np.sqrt(3), 2.0),
    'p3': (1.0,),
    'p4': (1.0,),
    'cm': (1.0, np.sqrt(2), np.sqrt(3), 2.0),
    'cmm': (1.0, np.sqrt(2), np.sqrt(3), 2.0),
    'p6': (1.0,),
    'p3m1': (1.0,),
    'pmm': (1.0, np.sqrt(2), np.sqrt(3), 2.0),
    'p4g': (1.0,),
    'pmg': (1.0, np.sqrt(2), np.sqrt(3), 2.0),
    'pm': (1.0, np.sqrt(2), np.sqrt(3), 2.0),
    'pgg': (1.0, np.sqrt(2), np.sqrt(3), 2.0),
    'pg': (1.0, np.sqrt(2), np.sqrt(3), 2.0),
    'p31m': (1.0,),
    'p4m': (1.0,),
    'p6m': (2.0,)
}

# Recommended angles for sampling.  Where angles are fixed (all except p1 and p2),
# they will be ignored.
RECOMMENDED_ANGLES = {
    'p1': (np.pi/2.0, np.pi/3.0, np.pi/4.0, np.pi/6.0),
    'p2': (np.pi/2.0, np.pi/3.0, np.pi/4.0, np.pi/6.0),
    'p3': (np.pi/3.0,),
    'p4': (np.pi/2.0,),
    'cm': (np.pi/2.0,),
    'cmm': (np.pi/2.0,),
    'p6': (2.0*np.pi/3.0,),
    'p3m1': (np.pi/3.0,),
    'pmm': (np.pi/2.0,),
    'p4g': (np.pi/2.0,),
    'pmg': (np.pi/2.0,),
    'pm': (np.pi/2.0,),
    'pgg': (np.pi/2.0,),
    'pg': (np.pi/2.0,),
    'p31m': (2.0*np.pi/3.0,),
    'p4m': (np.pi/2.0,),
    'p6m': (np.pi/3.0,)
}

def _get_generator(stoich, group, Nx, angle, length_ratio, count=False, idx=None):
    r"""
    Produce a generator for each stoichiometry, group, etc.

    Parameters
    ----------
    stoich : tuple
        (X:Y) stoichiometry of system.
    group : str
        Hermann-Mauguin name of wallpaper group of interest.
    Nx : int
        (Smallest) number of grid points to use on the group's fundamental domain.
    angle : tuple
        Angle (in radians) to sample if angle is variable for group's fundamental domain.
    length_ratio : tuple
        Length ratio of sides to sample if lengths are variable for group's fundamental domain.
    count : bool
        Whether or not to just count the number of total configurations for a group.
    idx : int
        Solution index to use if not just counting.

    Returns
    -------
    generator
        Generator for the primitive cell for each group, stoichiometry, etc.
    """

    params = dict(stoichiometry=stoich, \
        sample_groups=(wallpaper.WallpaperGroup(name=group),), \
        count_configurations=count, \
        log_level=0, # Do not need any output so suppress everything
        place_min=1, place_max=None, \
        angles=angle, length_ratios=length_ratio, \
        grid_count=Nx, \
        random_seed=0, \
        weighting_exponent=0, # Equal weighting, good for exhaustion
        merge_sets=False, \
        tolerance=1.0e-6, \
        debug=False, \
        chosen_solution_idx=idx, \
        sample_count=None
        )
    x = wallpaper.generate_wallpaper(**params)

    return x

def _exhaust(stoich, group, Ng, angle, length_ratio, congruent, idx=None):
    r"""
    Create and exhaust a generator of configurations for a specific group and solution index.

    Parameters
    ----------
    stoich : tuple
        (X:Y, ...) stoichiometry of system.
    group : str
        Hermann-Mauguin name of wallpaper group of interest.
    Ng : int
        Number of grid points to use.
    angle : tuple
        Angle (in radians) to sample if angle is variable for group's fundamental domain.
    length_ratio : tuple
        Length ratio of sides to sample if lengths are variable for group's fundamental domain.
    congruent : bool
        Whether or not the structural ensemble should be made "congruent" with a
        p1 equivalent cell in terms of its node density.
    idx : int
        Solution index to use if not just counting.

    Returns
    -------
    tuple
        (solution index, \# of realizations, 1x1 primitive cell of realizations, Nx)
    """

    nx = wallpaper.Nx(Ng, group, length_ratio[0]) if congruent else Ng
    x = _get_generator(stoich, group, nx, angle, length_ratio, count=False, idx=idx)
    res = [b for a, b in enumerate(x)]

    return (idx, len(res), res, nx)

class EnsembleFilter:
    r"""
    Filter **all** configurations (different solutions and their realizations) of chosen
    wallpaper groups.  This does **not** imply each configuration is truly unique, though.
    An ensemble is considered "congruent" if each group's
    grid density has been tuned to match an equivalent p1 primitive cell.
    If they are not congruent, each group's fundamental domain uses the Ng
    specified to produce a grid.

    Notes
    -----
    Be cautious when specifying parameters as this will seek to exhaustively search
    all possible configurations (solutions and their realizations) which can become
    a very large number if a large grid is used, for example.

    Parameters
    ----------
    congruent : bool
        Whether or not to make groups congruent with a p1 reference cell.
    """

    def __init__(self, congruent=True):
        self.__congruent = congruent

    def _get(self, group=None, stoich=None, Ng=None, angle=(np.pi/2.,), length_ratio=(1.0,), cores=1, max_configs=np.inf):
        r"""
        For a given stoichiometry, group, etc. generate all lattices corresponding to all realizations of all "solutions" to the CSP defining the system.

        Parameters
        ----------
        group : str
            Hermann-Mauguin name of wallpaper group of interest.
        stoich : tuple
            (X:Y, ...) stoichiometry of system.
        Ng : int
            Number of grid points to use.
        angle : tuple
            Angle (in radians) to sample if angle is variable for group's fundamental domain.
        length_ratio : tuple
            Length ratio of sides to sample if lengths are variable for group's fundamental domain.
        cores : int
            Number of CPU cores to devote to this.
        max_configs : int, float
            Max number of total configurations allowable - if more than this is found, the group is skipped.

        Returns
        -------
        res : array(tuple)
            Array of [(solution index, # of realizations, 1x1 primitive cells, Nx)] for all unique solutions and realizations thereof.
        """

        # Count total configurations
        nx_ = wallpaper.Nx(Ng, group, length_ratio[0]) if self.__congruent else Ng
        try:
            x = _get_generator(stoich, group, nx_, angle, length_ratio, count=True, idx=None)
            N_configs = next(x)
        except (RuntimeError, RuntimeWarning, Warning, Exception, ValueError) as e:
            raise Exception("error counting number of configurations: {}".format(e))

        if (N_configs >= max_configs): raise Exception('N_configs = {} > max_configs = {}'.format(N_configs, max_configs))

        # From configurations, infer the number of solutions possible by trying all
        N_sols = 0
        for i in range(N_configs):
            try:
                x = _get_generator(stoich, group, nx_, angle, length_ratio, count=False, idx=i)
                next(x)
            except ValueError as e:
                if ("invalid chosen_solution_idx" in str(e)):
                    break
                else:
                    raise Exception("unexpected ValueError: {}".format(e))
            except Exception as e:
                raise Exception("unexpected Exception: {}".format(e))
            else:
                N_sols += 1

        # Generate all configurations in a parallel manner
        if (cores != 1):
            res = [[]]*N_sols
            def _callback(x):
                res[x[0]] = x

            pool = Pool(cores)
            multi_res = [pool.apply_async(_exhaust, (stoich, group, Ng, angle, length_ratio, self.__congruent, i_), callback=_callback) for i_ in range(N_sols)]
            pool.close()
            pool.join()

            # Block and propagate any errors
            for mr in multi_res:
                try:
                    i, l, r, nx = mr.get()
                except Exception as e:
                    raise Exception('error on group {}: {}'.format(group, e))

            pool.terminate()
        else:
            res = [_exhaust(stoich, group, Ng, angle, length_ratio, self.__congruent, idx=i_) for i_ in range(N_sols)]

        return res

    def filter(self, groups, stoich, Ng, potentials, distance, count,
        angles=(np.pi/2.0,), length_ratios=(1.0,), cores=1, radii=None, callback=None):
        r"""
        Filter **all** configurations from a set based on energy and provide
        a generator to those configurations.

        Parameters
        ----------
        groups : array
            Hermann-Mauguin names of wallpaper groups of interest.
        stoich : tuple
            (X:Y, ...) stoichiometry of system.
        Ng : int
            Number of grid points to use.
        potentials : dict(tuple(int or str), paccs.potential.Potential)
            For energy evaluation.  See :py:func:`paccs.crystal.CellTools.energy`.
        distance : float
            For energy evaluation.  See :py:func:`paccs.crystal.CellTools.energy`.
        count : int
            The **total** number of cells to allow through.  The cells with the lowest energies
            will be yielded.
        angles : tuple
            Angles (in radians) to sample if angle is variable for group's fundamental domain.
        length_ratios : tuple
            Length ratios of sides to sample if lengths are variable for group's fundamental domain.
        cores : int
            Number of CPU cores to devote to this.
        radii: tuple(float)
            The radii of atom types.  If specified, performs automatic rescaling to contact.
            See :py:func:`paccs.minimization.filter`.
        callback : callable
            Callback function for :py:func:`paccs.minimization.filter`. This can be used
            to access the energies of these filtered configurations.

        Returns
        -------
        generator(tuple(paccs.wallpaper.WallpaperGroup, paccs.crystal.Cell))
            A generator yielding cells in the same manner as
            :py:func:`paccs.wallpaper.generate_wallpaper`, only filtered.

        Example
        -------
        >>> allow_to_pass = 100
        >>> ens = ensemble.EnsembleFilter(congruent=True)
        >>> cback = ensemble.EnergyHistogramCallback(count=allow_to_pass)
        >>> generator = ens.filter(['p1', 'p2', 'p3'], (1,2), 6, potentials, distance, allow_to_pass,
            angles=(numpy.pi/2.0,), length_ratios=(1.0,), cores=1, radii=(0.5, 0.5), callback=cback)
        >>> print cback.below_threshold # retreive the energy of the lower 100 structures
        >>> plt.hist(cback.below_threshold, bins='auto') # histogram the results
        """

        # Get all the unique configurations for all groups and ratios.
        total_configs = []
        for g,l,a in [(g_,l_,a_) for g_ in groups for l_ in length_ratios for a_ in angles]:
            sols = self._get(group=g, stoich=stoich, Ng=Ng, angle=(a,), length_ratio=(l,), cores=cores, max_configs=np.inf)
            for i in range(len(sols)):
                total_configs += sols[i][2]
        generator = (c for c in total_configs)

        return minimization.filter(generator, potentials, distance, count, radii, callback)

class EnergyHistogramCallback:
    r"""
    A class with an example callback function for use with filtering to view how the energies were sorted.

    Parameters
    ----------
    count : int
        Number of (lowest energy) configurations to allow through.
    """

    def __init__(self, count):
        self.__energies = {'below_threshold':None, 'above_threshold':None}
        self.__count = count

    def __call__(self, filter_energies):
        r"""
        Stores the filtered and sorted energies internally when this class is called.

        Parameters
        ----------
        filter_energies : array
            Array of sorted energies (lowest to highest).
        """

        self.__energies['below_threshold'] = filter_energies[:self.__count]
        self.__energies['above_threshold'] = filter_energies[self.__count:]

    @property
    def below_threshold(self):
        r"""
        Returns
        -------
        array
            Sorted array of energy values for all configurations allowed to pass the filter.
        """

        return self.__energies['below_threshold']

    @property
    def above_threshold(self):
        r"""
        Returns
        -------
        array
            Sorted array of energy values for all configurations that did not pass the filter.
        """

        return self.__energies['above_threshold']

if __name__ == "__main__":
    print(__file__)
