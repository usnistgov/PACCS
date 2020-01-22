"""
Systematically enumerate lattice configurations.
"""

import os
import copy
import numpy as np
import itertools
import cython
cimport cython
cimport libc.math
cimport numpy as np

from collections import Counter
from itertools import chain
from numpy import ndarray
from numpy cimport ndarray

# String to indicate a vacancy on the lattice
__EMPTY_SITE__ = "__EMPTY__"

class EnumeratedLattice(object):
    """
    Systematically enumerate all possible configurations on a lattice.
    Recipe does not need to contain all lattice site types in chunks, but chunks
    must contain all possible lattice site types.

    Although the input data types are general, the output of :py:func:`get()`
    and other routines which call it (such as the generators produced by various
    members of this class), the atom types will always be turned into strings
    because they are part of an ndarray of strings.  This may mean type
    conversion is necessary later, especially if the atom types were input as
    integer indices, for example.

    Parameters
    ----------
    chunks : dict
        Dictionary with key/value pairs of (lattice site type : number available). For example: {"corners": 4, "edges": 10, "faces": 50, "midpoints":8}.
    recipe : dict
        Dictionary with key/value pairs of (lattice site type : dict(atom types : number)). For example: {"corners":{"A":2, "B":1}, "edges":{"B":1, "D":2}, "faces":{"A":3, "C":3}}.
    """

    def __init__(self, chunks, recipe):
        if (not isinstance(chunks, dict)): raise TypeError("chunks should be a dict")
        if (not isinstance(recipe, dict)): raise TypeError("recipe should be a dict")
        if (np.any([not k in chunks for k in recipe])): raise Exception("all lattice site types in recipe must be specified in chunks")
        for key in recipe.keys():
            total = np.sum([recipe[key][x] for x in recipe[key]])
            if total > chunks[key]: raise ValueError("too many particles in recipe for chunks to accommodate")
            if (np.any([x == __EMPTY_SITE__ for x in recipe[key]])): raise ValueError("cannot name atoms \"{}\" as it conflicts with empty position designation".format(__EMPTY_SITE__))

        self.__chunks = chunks
        self.__recipe = recipe
        ctr = Counter(chain.from_iterable(self.__recipe[a].keys() for a in self.__recipe))
        self.__atom_types = tuple(sorted(ctr.keys()))
        self.__natom_types = len(self.__atom_types)
        self.__rng = False
        self.__used = {}
        self._enum()

    def all(self):
        """
        Systematically loop through all solutions to the recipe in order.
        This uses a clone of the calling object and resets the "memory" of any
        configurations returned thus far.  Therefore, the iterator will always
        return all configurations from a "fresh" object.

        Returns
        -------
        generator
            Generator that systematically returns configurations.
        """

        clone = copy.deepcopy(self)
        clone.__used = {}

        return (clone.get(idx, repeat=False) for idx in range(clone.__nconfigs))

    def ran(self, seed=0, clone=True):
        """
        Systematically loop through all solutions to the recipe in a random order.
        This uses a clone of the calling object and resets the *memory* of any
        configurations returned thus far.  Therefore, the iterator will always
        return all configurations from a *fresh* object and stops once all unique
        configurations have been looped over.

        Parameters
        ----------
        seed : int
            Seed for random generation of sequence.
        clone : bool
            By default, a clone of this object used internally with no previously visited
            configurations to maintain the above discussed behavior; however,
            this requires a deepcopy() operation which can be expensive if it is
            unneccesary. If clone is set to False, then this object itself is used **in
            its current state** and will be exhausted by the end of the generator which is returned.
            Notably, this will not reset any configurations already visited which will not be
            revisited in this case.

        Returns
        -------
        generator
            Generator that returns unique configurations in an random order.
        """

        if (clone):
            cc = copy.deepcopy(self)
            cc.__used = {}
        else:
            cc = self

        return (cc.random(maxiter=-1, seed=seed) for idx in range(cc.__nconfigs))

    def get(self, idx, repeat=False):
        """
        Return the configuration corresponding to a specific index.
        This reconstructs a configuration based on its solution index in the
        hypercube, and tests if it has been used before. If repeat is set to False, then an Exception will be raised if an old configuration is chosen.

        Parameters
        ----------
        idx : int
            Base 10 (unrolled) index in the hypercube defining recipe's solutions to return.
        repeat : bool
            Whether or not to allow configurations to be repeated. (default=False)

        Returns
        -------
        dict
            Dictionary of (lattice site type : ndarray of atom types, as strings, at each lattice site for this type).
        """

        # Check if idx has been requested before
        if (idx in self.__used):
            if (repeat):
                self.__used[idx] += 1
            else:
                raise Exception("configuration {} has been requested before".format(idx))
        else:
            self.__used[idx] = 1

        return self._make_config(self._get_address(idx))

    def random(self, maxiter=-1, seed=0):
        """
        Generate a new, (generally) unique random structure.

        Parameters
        ----------
        maxiter : int
            Maximum number of attempts to find a new random structure before a
            randomly chosen old one is returned.  If negative, there is no max
            used. A value of zero will cause the code to choose a configuration
            randomly without guaranteeing uniqueness. (default=-1)
        seed : int
            Random number generator seed. Only used the first time this routine
            is called. (default=0)

        Returns
        -------
        dict
            Dictionary of (lattice site type : ndarray of atom types at each
            lattice site for this type).
        """

        # Set RNG the first time
        if (not self.__rng):
            self.__rng = np.random.RandomState(seed)

        # Set upper bound for iteration
        cdef long int upper = 0
        if (maxiter < 0):
            upper = self.__nconfigs
        else:
            upper = maxiter

        # Try to get a new structure
        cdef long int rand = self.__rng.randint(0, self.__nconfigs)
        cdef long int niter = 0
        while (niter < upper):
            niter += 1
            try:
                config = self.get(rand, repeat=False)
            except:
                # Walk systematically away until next new position found
                rand -= 1
                if (rand < 0):
                    rand = self.__nconfigs -1

                # Just blindly choose randomly
                # rand = self.__rng.randint(0, self.__nconfigs)
            else:
                return config

        # If reached the max attempts (or sampling randomly allowing replacement), just take the last guess
        return self.get(rand, repeat=True)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def _enum(self):
        """
        Perform the systematic enumeration of all configuration.
        """

        """
        Assume "chunks" used in the recipe are arbitrarily ordered
        [ ... chunk 1 ... ][ ... chunk 2 ... ] ... [ ... chunk n ...]

        Then within each used chunk, we have n_C_k ways to place each species on the lattice sites in each chunk. So if chunk 1 = "faces" and there are 20 sites, onto which we place 5 "A", 6 "B", and 2 "C" then

        [ ... chunk 1 ... ] = 20_C_5 * 15_C_6 * 9_C_2 = 20!/(5! * 6! * 2! * 7!)

        Each n_C_k can be thought of as the edge of hypercube, which has as many dimensions as n_C_k operations as are in all the chunks put together.

        """

        cdef long int n, k

        # Iterate over each chunk used in recipe, sort from rarest to most common
        seq = []
        self.__hyperedge = []
        for chunk in sorted(self.__recipe, key=lambda x: self.__chunks[x]):
            pattern = []
            n = self.__chunks[chunk]
            for atom_type in sorted(self.__recipe[chunk].keys()):
                k = self.__recipe[chunk][atom_type]
                res = list(itertools.combinations(range(n), k)) # (0,N-1) to convert integer to array index
                self.__hyperedge.append(len(res))
                pattern.append((atom_type, res))
                n -= k
            seq.append((chunk, pattern))
        self.__hyperedge = np.array(self.__hyperedge, dtype=np.float32)

        # Sequence defines the relative placement of atoms of each type in each position
        self.__sequence = seq
        self.__nconfigs = int(np.prod(self.__hyperedge))

    @property
    def nconfigs(self):
        """
        Returns
        -------
        int
            Number of configurations.
        """

        return self.__nconfigs

    @cython.boundscheck(False)
    def _make_config(self, address):
        """
        Create a configuration from the chosen indices of the hypercube.

        Parameters
        ----------
        address : array(int)
            Array of indices corresponding to self.__sequence.

        Returns
        -------
        dict
            Dictionary of (lattice site type : ndarray of atom types at each lattice site for this type as strings).
        """

        cdef long int lidx, aidx, ctr

        __sequence = self.__sequence
        __chunks = self.__chunks
        _fill_open = self._fill_open

        config = {}

        ctr = 0
        for lidx in range(len(__sequence)):
            lattice_type = __sequence[lidx][0]
            chunk_instruct = [__EMPTY_SITE__]*__chunks[lattice_type]
            for aidx in range(len(__sequence[lidx][1])):
                atom_type = __sequence[lidx][1][aidx][0]

                # Get empty position indices
                pos = __sequence[lidx][1][aidx][1][address[ctr]]
                ctr += 1

                # Fill specified empty positions in this chunk
                _fill_open(chunk_instruct, atom_type, pos)

            config[lattice_type] = chunk_instruct

        return config

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def _fill_open(self, instruct, val, pos):
        """
        Fill the open positions on a lattice.

        Parameters
        ----------
        instruct : array
            ndarray of strings indicating the atom type at each position.
        val : str or int
            Atom type to place.
        pos : array
            Array of open position indices to place at.
        """

        cdef long int ctr = -1
        cdef long int counts = 0
        cdef long int start = 0, end = len(instruct)
        cdef long int p = 0, idx = 0

        for p in sorted(pos):
            for idx in range(start, end):
                if (instruct[idx] == __EMPTY_SITE__):
                    ctr += 1
                    if (ctr == p):
                        instruct[idx] = val
                        counts += 1
                        start = idx+1
                        break

        # Sanity check
        if (counts != len(pos)): raise Exception("unable to place all {}".format(val))

    def _get_address(self, idx):
        """
        From a base 10 integer convert to hypercube indices indicating the "address" of the configuration.

        Parameters
        ----------
        idx : int,double
            Base 10 integer/float index of configuration to convert to hypercube indices.

        Returns
        -------
        ndarray(numpy.uint64)
            Set of instructions to use with self.__sequence that will define a configuration.
        """

        if (idx >= self.__nconfigs): raise ValueError("index requested is greater than the number of configurations possible")
        return _cython_get_address(idx, self.__hyperedge)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef np.ndarray[np.int32_t, ndim=1] _cython_get_address(double idx_, np.ndarray[np.float32_t, ndim=1] __hyperedge):
    """
    From a base 10 number convert to hypercube indices indicating the "address" of the configuration.

    Parameters
    ----------
    idx : double
        Base 10 index of configuration to convert to hypercube indices.

    Returns
    -------
    ndarray(numpy.uint64)
        Set of instructions to use with self.__sequence that will define a configuration.
    """

    cdef long int len_h = len(__hyperedge)
    cdef np.ndarray[np.uint64_t, ndim=1] address = np.zeros(len_h, dtype=np.uint64)
    cdef double div = 1, coord = 0
    cdef int i
    for i in range(len_h-1):
        div *= __hyperedge[i]

    for i in range(len_h):
        coord = libc.math.floor(idx_/div)
        idx_ -= coord*div
        address[len_h-1-i] = int(coord)
        if (i < len_h-1):
            div /= __hyperedge[len_h-2-i]

    return address
