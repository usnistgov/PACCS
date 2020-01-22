"""
Contains various types of pair potentials as well as an interface for the creation
of user-defined potentials.
"""

import cython
import numpy
import scipy.interpolate
import scipy.optimize
import warnings
cimport cython
cimport libc.math
cimport numpy

cdef class Potential:
    """
    Represents an arbitrary pairwise potential function.  Creating an instance of the
    :py:class:`Potential` class directly yields a potential
    which evaluates to zero everywhere.
    """

    cpdef (double, double) evaluate(self, double r):
        """
        Evaluates the potential function for a given pairwise separation distance.

        Parameters
        ----------
        r : cython.double
            The separation distance at which to evaluate.

        Returns
        -------
        u, f : cython.double, cython.double
            The energy and force, respectively.  Negative forces correspond to
            pairwise attraction.
        """

        return 0.0, 0.0

    @cython.boundscheck(False)
    @cython.cdivision(True)
    @cython.wraparound(False)
    def evaluate_array(self, numpy.ndarray[numpy.float64_t, ndim=1] r):
        """
        Evaluates the potential function for an array of pairwise separation distances.

        Parameters
        ----------
        r : numpy.ndarray
            A one-dimensional array of separation distances at which to evaluate.

        Returns
        -------
        u, f : numpy.ndarray, numpy.ndarray
            The energy and force, respectively.  These arrays will have the same
            shape as the distance array provided.
        """

        cdef numpy.ndarray[numpy.float64_t, ndim=1] u = numpy.zeros_like(r, dtype=numpy.float64)
        cdef numpy.ndarray[numpy.float64_t, ndim=1] f = numpy.zeros_like(r, dtype=numpy.float64)

        cdef long index
        for index in range(r.shape[0]):
            u[index], f[index] = self.evaluate(r[index])

        return u, f

    def __reduce__(self):
        return self.__class__, ()

@cython.final(True)
cdef class Transform(Potential):
    ur"""
    Permits for arbitrary shifting and scaling to be applied to a
    predefined pairwise potential.

    Parameters
    ----------
    potential : Potential
        The predefined potential.
    sigma : float
        The length scale factor.
    epsilon : float
        The energy scale factor.
    s : float
        The length shift factor.
    phi : float
        The energy shift factor.

    Notes
    -----
    If the given potential is :math:`U'(r')`, then the resulting potential is of the form:

    .. math:: U(r)=\epsilon\left[U'\left(\frac{r}{\sigma}-s\right)+\phi\right]

    With the default options, no transformations are performed.
    """

    cdef Potential __potential
    cdef double __sigma
    cdef double __epsilon
    cdef double __s
    cdef double __phi

    cdef double __prefactor

    def __init__(self, potential, sigma=1.0, epsilon=1.0, s=0.0, phi=0.0):
        self.__potential = potential
        self.__sigma = sigma
        self.__epsilon = epsilon
        self.__s = s
        self.__phi = phi

        self.__prefactor = self.__epsilon / self.__sigma

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef inline (double, double) evaluate(self, double r):
        """
        See :py:func:`Potential.evaluate`.
        """

        cdef double u
        cdef double f

        u, f = self.__potential.evaluate((r / self.__sigma) - self.__s)
        return self.__epsilon * (u + self.__phi), self.__prefactor * f

    def __reduce__(self):
        return self.__class__, (self.__potential, self.__sigma, self.__epsilon, self.__s, self.__phi)

@cython.final(True)
cdef class Piecewise(Potential):
    """
    Permits for two potentials to be joined together and evaluated
    individually depending on the given pairwise separation distance.

    Parameters
    ----------
    near_potential : Potential
        The near-range interaction potential.
    far_potential : Potential
        The far-range interaction potential.
    rc : double
        The cutoff which determines whether the near-range or far-range
        potential will be evaluated.  At this distance exactly, the
        far-range potential will be selected.
    """

    cdef Potential __near_potential
    cdef Potential __far_potential
    cdef double __rc

    def __init__(self, near_potential, far_potential, rc):
        self.__near_potential = near_potential
        self.__far_potential = far_potential
        self.__rc = rc

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef inline (double, double) evaluate(self, double r):
        """
        See :py:func:`Potential.evaluate`.
        """

        if r < self.__rc:
            return self.__near_potential.evaluate(r)
        else:
            return self.__far_potential.evaluate(r)

    def __reduce__(self):
        return self.__class__, (self.__near_potential, self.__far_potential, self.__rc)

@cython.final(True)
cdef class DNACC(Potential):
    ur"""
    Represents a pair potential between two DNA-coated colloids whose interaction
    is given by the self-consistent mean field theory in Varilly et al., *J. Chem. Phys.* **137** (2012).
    This assumes the DNA grafts are given by rigid rods, and the Derjaguin approximation is used
    to compute potentials between curved surfaces.

    Parameters
    ----------
    r1 : float
        Radius of first sphere.
    r2 : float
        Radius of second sphere.
    lengths : dict
        Dictionary of {name (str) : length (float)}. This must contain all constructs used in the system.
    sigma1 : dict
        Dictionary of {name (str) : grafting density (float)} on sphere 1.
    sigma2 : dict
        Dictionary of {name (str) : grafting density (float)} on sphere 2.
    beta_DeltaG0 : dict
        Dictionary of all {tuple(name (str), name (str)): :math:`\Delta G_{\rm bind} / k_{\rm B}T` (float) } for all construct pairs.
    beta : double
        (optional) :math:`1/k_{\rm B}T` of the system, assumed to be unity if it is not specified.

    Notes
    -----
    This implicitly works with units of nm. This theory only provides a prediction up to the surface-to-surface contact point,
    so to make this continuous a steep repulsive wall potential has been added of the form:

    .. math:: U_{\rm w}(r) = \epsilon_{\rm w} \left( \frac{r_{\rm min}}{r + r_{\rm min}\left(\epsilon_{\rm w}^{1/n} - 1\right) } \right)^n - \epsilon_{\rm w} + V(h_{\rm min})

    where the quantity :math:`\beta V` is predicted by the mean field theory, and :math:`r_{\rm min}` corresponds to the smallest :math:`h` (surface separation)
    this is computed for (:math:`r_{\rm min} = r1 + r2 + h_{\rm min}`).  Internally, :math:`n = 50` and :math:`\epsilon_{\rm w} = 1`.

    When :math:`r < r_{\rm min}` the wall potential is computed directly, whereas when :math:`r \ge r_{\rm min}` the
    DNACC code's potential is interpolated to predict the pair interaction and the attractive force between a pair of colloids.
    """

    cdef int __hbins
    cdef double __rcut
    cdef double __shift
    cdef double __walle
    cdef double __walln
    cdef double __hmin
    cdef double __hmax
    cdef double __Vmin

    cdef double __r1
    cdef double __r2
    cdef double __beta
    cdef dict __lengths
    cdef dict __sigma1
    cdef dict __sigma2
    cdef dict __beta_DeltaG0

    cdef numpy.float64_t[:] __h
    cdef numpy.float64_t[:] __V
    cdef numpy.float64_t[:] __t
    cdef numpy.float64_t[:] __c
    cdef int __k

    def __init__(self, r1=0, r2=0, lengths={}, sigma1={}, sigma2={}, beta_DeltaG0={}, beta=1.0):
        try:
            import dnacc
            from dnacc.units import nm
        except:
            raise Exception("cannot locate DNACC library, check that it is installed and in your $PYTHONPATH")

        # Value checks
        self.__hbins = 1000
        self.__r1 = r1*nm
        self.__r2 = r2*nm
        self.__lengths = lengths
        self.__sigma1 = sigma1
        self.__sigma2 = sigma2
        self.__beta_DeltaG0 = beta_DeltaG0
        self.__hmax = 2.0*numpy.max([self.__lengths[t] for t in self.__lengths])*nm + 1.0*nm
        self.__hmin = self.__hmax/1000.
        self.__rcut = self.__hmax + self.__r1 + self.__r2
        self.__beta = beta

        # Steep repulsive wall at contact
        self.__walle = 1.0
        self.__walln = 50.0
        self.__shift = (self.__r1 + self.__r2 + self.__hmin)*(self.__walle**(1./self.__walln)) - (self.__r1 + self.__r2 + self.__hmin)
        assert (self.__shift >= 0)

        if (self.__r1 <= 0): raise ValueError("radius_1 must be positive")
        if (self.__r2 <= 0): raise ValueError("radius_2 must be positive")
        if (numpy.any([self.__lengths[t] for t in self.__lengths]) <= 0):
            raise ValueError("all lengths must be positive")
        if (numpy.any([self.__sigma1[t] < 0 for t in self.__sigma1])):
            raise ValueError("all grafting densities on sphere 1 must be >= 0")
        if (numpy.any([self.__sigma2[t] < 0 for t in self.__sigma2])):
            raise ValueError("all grafting densities on sphere 2 must be >= 0")
        if (numpy.any([k not in self.__lengths for k in self.__sigma1])):
            raise Exception("grafting densities on sphere 1 contains unknown constructs")
        if (numpy.any([k not in self.__lengths for k in self.__sigma2])):
            raise Exception("grafting densities on sphere 2 contains unknown constructs")
        if (self.__beta <= 0.0): raise ValueError("beta must be positive")

        spec = {}
        for (i,j) in self.__beta_DeltaG0:
            spec[(i,j)] = 0
            spec[(j,i)] = 0

        for (i,j) in self.__beta_DeltaG0:
            if (j,i) in self.__beta_DeltaG0:
                if (self.__beta_DeltaG0[(i,j)] != self.__beta_DeltaG0[(j,i)]): raise ValueError("binding affinities are not symmetric")
            spec[(i,j)] = 1
            spec[(j,i)] = 1
            if (i not in set([k for k in self.__lengths])):
                raise Exception("unrecognized construct {}".format(i))
            if (j not in set([k for k in self.__lengths])):
                raise Exception("unrecognized construct {}".format(j))
        if (numpy.any(spec == 0)): raise Exception("missing binding affinities")
        if (len(spec) != len(self.__lengths)**2): raise Exception("missing binding affinities")

        # Compute V/kT(h)
        plates = dnacc.PlatesMeanField()

        for name in lengths:
            if (name in self.__sigma1):
                global_idx = plates.add_tether_type(plate='1',
                    sticky_end=name,
                    L=self.__lengths[name]*nm,
                    sigma=self.__sigma1[name]/(nm**2))

            if (name in self.__sigma2):
                global_idx = plates.add_tether_type(plate='2',
                    sticky_end=name,
                    L=self.__lengths[name]*nm,
                    sigma=self.__sigma2[name]/(nm**2))

        for (name1, name2) in self.__beta_DeltaG0:
            plates.beta_DeltaG0[name1, name2] = self.__beta_DeltaG0[(name1, name2)]

        plates.at(self.__hmax).set_reference_now()
        self.__h = numpy.linspace(self.__hmin, self.__hmax, self.__hbins)
        V_plate_arr = [plates.at(h).free_energy_density for h in self.__h]
        self.__V = dnacc.calc_spheres_potential(self.__h, V_plate_arr, R1=self.__r1, R2=self.__r2)
        self.__Vmin = self.__V[0]

        try:
            self.__t, self.__c, self.__k = scipy.interpolate.splrep(self.__h, self.__V, s=0, k=3)
        except:
            raise Exception("unable to find spline for DNACC potential")

    def __pnames__(self):
        """
        Get the names of the input variables.
        """

        return ("r1", "r2", "lengths", "sigma1", "sigma2", "beta_DeltaG0", "beta")

    def __reduce__(self):
        return self.__class__, (self.__r1, self.__r2, self.__lengths, self.__sigma1, self.__sigma2, self.__beta_DeltaG0, self.__beta)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef inline (double, double) evaluate(self, double r):
        """
        See :py:func:`Potential.evaluate`.
        """

        cdef double u = 0.0
        cdef double f = 0.0
        cdef double x = 0.0
        cdef double y = 0.0
        cdef double h = 0.0

        if (r < self.__hmin + self.__r1 + self.__r2):
            # Return "shifted" WCA potential as spheres approach contact point (actually last point from theory, h ~ 0, but h != 0).
            y = r + self.__shift
            x = self.__walle*((self.__r1 + self.__r2 + self.__hmin)/y)**self.__walln
            u = x - self.__walle + self.__Vmin/self.__beta
            f = self.__walln*x/y
        elif (r < self.__rcut):
            # Interpolate potential and numerically estimate derivative.
            # Theory predicts beta*U, so divide by beta to get just energy and force.
            h = r - self.__r1 - self.__r2
            u = scipy.interpolate.splev([h], (self.__t, self.__c, self.__k), der=0)/self.__beta
            f = -scipy.interpolate.splev([h], (self.__t, self.__c, self.__k), der=1)/self.__beta
        else:
            u = 0.0
            f = 0.0

        return u, f

@cython.final(True)
cdef class SquareWell(Potential):
    ur"""
    Represents a square-well potential.

    .. math:: U(r) = \left\{
                    \begin{array}{cl}
                    0 & r \ge \lambda \sigma \\
                    -\epsilon & \sigma \le r < \lambda \sigma \\
                    +\infty & r < \sigma
                    \end{array}
                    \right.

    Parameters
    ----------
    epsilon : float
        Energy well depth.
    sigma : float
        Inner edge of energy well.
    lambda\_ : float
        The outer edge of the energy well occurs at :math:`\lambda \sigma`.

    Notes
    -----
    **Importantly**, the derivative (force) will be returned as being zero **everywhere** because the potential is discontinuous except exactly at the the edges of the wells where it is :math:`\pm \infty`.  To protect numerical stability, this is simply (erroneously) disregarded.  This means optimizations using conjugate gradients, or any derivative of the potential, will not function as expected.  It is still possible to use this potential for stochastic optimization processes and other calculations.
    """

    cdef double __sigma
    cdef double __epsilon
    cdef double __lambda
    cdef double __rc

    def __init__(self, sigma=1.0, epsilon=1.0, lambda_=1.0):
        self.__sigma = sigma
        self.__epsilon = epsilon
        self.__lambda = lambda_
        self.__rc = self.__sigma*self.__lambda

    def __pnames__(self):
        """
        Get the names of the input variables.
        """

        return ("sigma", "epsilon", "lambda_")

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef inline (double, double) evaluate(self, double r):
        """
        See :py:func:`Potential.evaluate`.
        """

        if (r < self.__sigma):
            return numpy.inf, 0.0
        elif (r < self.__rc):
            return -self.__epsilon, 0.0
        else:
            return 0.0, 0.0

    def __reduce__(self):
        return self.__class__, (self.__sigma, self.__epsilon, self.__lambda)

@cython.final(True)
cdef class LennardJonesType(Potential):
    ur"""
    Represents a potential whose parameters can be set to yield Lennard-Jones and
    Lennard-Jones-like pairwise interactions.

    Parameters
    ----------
    sigma : float
        The length scale factor.
    epsilon : float
        The energy scale factor.
    lambda\_ : float
        A parameter controlling the shape of the potential.  When :math:`\lambda>0`, the
        potential contains a well; otherwise, it does not.
    n : float
        The exponent controlling how quickly the potential approaches 0 as the separation
        distance increases.
    s : float
        A parameter controlling the position of the minimum.  In the :math:`\lambda=1` case,
        it appears at :math:`r=s\sigma`.

    Notes
    -----
    The potential function is of the form:

    .. math:: U(r)=4\epsilon\left(\frac{1}{\nu^{2n}}-\frac{\lambda}{\nu^n}\right),\quad\nu=\frac{r}{\sigma}-s+2^{1/n}

    Setting :math:`\lambda=1`, :math:`n=6` and :math:`s=2^{1/n}` yields the standard
    6/12 Lennard-Jones potential.  These are the default parameters.
    """

    cdef double __sigma
    cdef double __epsilon
    cdef double __lambda
    cdef double __n
    cdef double __s

    cdef double __u_prefactor
    cdef double __f_prefactor
    cdef double __shift

    def __init__(self, sigma=1.0, epsilon=1.0, lambda_=1.0, n=6.0, s=2.0 ** (1.0 / 6.0)):
        self.__sigma = sigma
        self.__epsilon = epsilon
        self.__lambda = lambda_
        self.__n = n
        self.__s = s

        self.__u_prefactor = 4.0 * self.__epsilon
        self.__f_prefactor = self.__u_prefactor * self.__n / self.__sigma
        self.__shift = self.__s - (2.0 ** (1.0 / self.__n))

    def __pnames__(self):
        """
        Get the names of the input variables.
        """

        return ("sigma", "epsilon", "lambda_", "n", "s")

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef inline (double, double) evaluate(self, double r):
        """
        See :py:func:`Potential.evaluate`.
        """

        cdef double nu = (r / self.__sigma) - self.__shift
        cdef double nu_inv = 1.0 / nu
        cdef double nu_inv_n = nu_inv ** self.__n
        cdef double nu_inv_2n = nu_inv_n * nu_inv_n
        cdef double nu_inv_n_lam = self.__lambda * nu_inv_n

        return self.__u_prefactor * (nu_inv_2n - nu_inv_n_lam), \
            self.__f_prefactor * nu_inv * ((2 * nu_inv_2n) - nu_inv_n_lam)

    def __reduce__(self):
        return self.__class__, (self.__sigma, self.__epsilon, self.__lambda, self.__n, self.__s)

@cython.final(True)
cdef class JaglaType(Potential):
    ur"""
    Represents a Jagla-type pairwise interaction potential.

    Parameters
    ----------
    sigma : float
        The length scale factor for the power term of the potential.
    epsilon : float
        The energy scale factor for the power term of the potential.
    n : float
        The exponent for the power term of the potential.
    s : float
        The horizontal shifting factor for the power term of the potential.
    a0 : float
        The energy scale factor for the first exponential term of the potential.
    a1 : float
        The length scale factor for the first exponential term of the potential.
    a2 : float
        The horizontal shifting factor for the first exponential term of the potential.
    b0 : float
        The energy scale factor for the second exponential term of the potential.
    b1 : float
        The length scale factor for the second exponential term of the potential.
    b2 : float
        The horizontal shifting factor for the second exponential term of the potential.

    Notes
    -----
    The potential function is of the form:

    .. math:: U(r)=\epsilon{\left(\frac{\sigma}{r-s}\right)}^n+\frac{a_0}{1+\exp\left[a_1\left(r-a_2\right)\right]}-\frac{b_0}{1+\exp\left[b_1\left(r-b_2\right)\right]}

    The default parameters yield a potential with a very sharp well of unit depth
    at :math:`r\approx1.028`.  Other wells can be created (similar to this one but
    with different depths) using :py:func:`JaglaType.make` as well as via
    manual specification of the parameters.
    """

    cdef double __sigma
    cdef double __epsilon
    cdef double __n
    cdef double __s
    cdef double __a0
    cdef double __a1
    cdef double __a2
    cdef double __b0
    cdef double __b1
    cdef double __b2

    cdef double __u_prefactor
    cdef double __f_prefactor

    def __init__(self, sigma=0.2, epsilon=10.0, n=36.0, s=0.8, \
        a0=11.0346, a1=404.396, a2=1.0174094, b0=1.3218525572553343, b1=1044.5, b2=1.0305952):
        self.__sigma = sigma
        self.__epsilon = epsilon
        self.__n = n
        self.__s = s
        self.__a0 = a0
        self.__a1 = a1
        self.__a2 = a2
        self.__b0 = b0
        self.__b1 = b1
        self.__b2 = b2

        self.__u_prefactor = self.__epsilon * (self.__sigma ** self.__n)
        self.__f_prefactor = self.__n * self.__u_prefactor

    @cython.boundscheck(False)
    @cython.cdivision(True)
    cpdef inline (double, double) evaluate(self, double r):
        """
        See :py:func:`Potential.evaluate`.
        """

        cdef double rs_inv = 1.0 / (r - self.__s)
        cdef double rs_inv_n = rs_inv ** self.__n
        cdef double exp_a = libc.math.exp(self.__a1 * (r - self.__a2))
        cdef double exp_b = libc.math.exp(self.__b1 * (r - self.__b2))
        cdef double sigm_exp_a = 1.0 / (1.0 + exp_a)
        cdef double sigm_exp_b = 1.0 / (1.0 + exp_b)
        cdef double sh_sigm_exp_a = self.__a0 * sigm_exp_a
        cdef double sh_sigm_exp_b = self.__b0 * sigm_exp_b

        return (self.__u_prefactor * rs_inv_n) + sh_sigm_exp_a - sh_sigm_exp_b, \
            (self.__f_prefactor * rs_inv_n * rs_inv) + (self.__a1 * exp_a * sh_sigm_exp_a * sigm_exp_a) - \
            (self.__b1 * exp_b * sh_sigm_exp_b * sigm_exp_b)

    @staticmethod
    def make(energy):
        ur"""
        Determines Jagla-type potential coefficients yielding potentials with
        specified well depths.

        Parameters
        ----------
        energy : float
            The desired depth of the potential well at its minimum.

        Returns
        -------
        parameters, well_position : tuple, float
            The parameters :math:`\sigma`, :math:`\epsilon`, :math:`n`, :math:`s`,
            :math:`a_0`, :math:`a_1`, :math:`a_2`, :math:`b_0`, :math:`b_1`, and
            :math:`b_2` required to generate a Jagla-type pairwise potential with
            the desired well depth by calling :py:class:`JaglaType`.
            Note that the separation distance of the potential well is also provided.

        Raises
        ------
        RuntimeError
            An optimization algorithm failed and the parameters could not be
            determined.
        """

        # Define the fixed parameters for the Jagla system
        sigma, epsilon, n, s, a0, a1, a2, b1, b2 = \
            0.2, 10.0, 36.0, 0.8, 11.0346, 404.396, 1.0174094, 1044.5, 1.0305952

        # Define the solver residual function
        def residual(b0):
            objective = lambda r: JaglaType(sigma, epsilon, n, s, a0, a1, a2, b0, b1, b2).evaluate(r)[0]
            minimize_result = scipy.optimize.minimize_scalar(objective, method="bounded", \
                bounds=(1.0, 1.05), options={"xatol": 0.0})
            if not minimize_result.success:
                raise RuntimeError("convergence error during Jagla minimization: {}".format(minimize_result.message))
            return minimize_result.x, minimize_result.fun + energy

        # Solve for b0 and return the results
        b0 = scipy.optimize.brentq(lambda b0: residual(b0)[1], 0.0, 2.0)
        return (sigma, epsilon, n, s, a0, a1, a2, b0, b1, b2), residual(b0)[0]

    def __reduce__(self):
        return self.__class__, (self.__sigma, self.__epsilon, self.__n, self.__s, \
            self.__a0, self.__a1, self.__a2, self.__b0, self.__b1, self.__b2)

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cpdef tuple _evaluate_fast(object cell, dict potentials, double distance, double bin_width=0.0):
    """
    Performs rapid energy evaluation.  See :py:func:`paccs.crystal.CellTools.energy`
    for information pertaining to the first three parameters.

    Parameters
    ----------
    bin_width : float
        If specified, this routine will return RDFs in histogram format.  The
        histograms will have bins of the specified width.  The distance cutoff
        will not be extended even if a bin overruns it.

    Returns
    -------
    energy, partial_jacobian, histogram: float, numpy.ndarray, numpy.ndarray
        The energy per atom of the cell, followed by the derivatives of
        the energy (per atom) with respect to the position coordinates of each atom
        in sequence.  The list begins with the first coordinate of the first
        atom of the first atom type in the cell, and progresses through
        coordinates, then atoms, then atom types.  Additionally, if requested,
        RDFs in histogram format.  Distribution of pair distances between
        closest and next closest bins will be performed to improve behavior
        of histograms when the ratio of atomic spacing to bin width is integral.
    """

    # Extract useful information from cell object for later fast access
    cdef long dimensions = cell.dimensions
    cdef long atom_types = cell.atom_types
    cdef numpy.ndarray[numpy.uint64_t, ndim=1] atom_counts = numpy.array(cell.atom_counts, dtype=numpy.uint64)
    cdef long total_atom_count = numpy.sum(atom_counts)
    cdef numpy.ndarray[numpy.float64_t, ndim=2] vectors = cell.vectors

    # Build two-dimensional atom array and type index lookup table
    cdef numpy.ndarray[numpy.uint64_t, ndim=1] atom_type_lookup = numpy.zeros((atom_types), dtype=numpy.uint64)
    cdef numpy.ndarray[numpy.float64_t, ndim=2] atoms = numpy.zeros((total_atom_count, dimensions), dtype=numpy.float64)
    cdef long atom_index_1 = 0, atom_index_2, type_index
    cdef int tt
    if atom_types > 0:
        atom_type_lookup[0] = 0
    for type_index in range(1, atom_types):
        atom_type_lookup[type_index] = atom_type_lookup[type_index - 1] + atom_counts[type_index - 1]
    for type_index in range(atom_types):
        atoms[atom_type_lookup[type_index]:atom_type_lookup[type_index] + atom_counts[type_index]] = cell.atoms(type_index)

    # Build array of potentials from provided indices or strings
    null_potential = Potential()
    potential_array = [[null_potential] * atom_types for index in range(atom_types)]
    if potentials is not None:
        for pair in potentials:
            source_type, target_type = pair

            # Convert names to indices
            try:
                if not isinstance(source_type, int):
                    source_type = cell.index(source_type)
                if not isinstance(target_type, int):
                    target_type = cell.index(target_type)
            except:
                # If either of these names/identifiers aren't in the cell, skip
                continue

            # Check if these types in the potential dictionary are present in the cell
            for tt in [source_type, target_type]:
                if (tt < 0 or tt >= atom_types):
                    raise Exception('illegal type index {} for a cell with only {} atom types - specify potentials using atom string names if you wish to provide potential information for cells missing certain atom types'.format(tt, atom_types))

            # Do the potential assignment
            if potential_array[source_type][target_type] is not null_potential:
                raise ValueError("duplicate pair potential assignment encountered")
            potential_array[source_type][target_type] = potentials[pair]
            if source_type != target_type:
                potential_array[target_type][source_type] = potentials[pair]
    cdef Potential[:, :] potential_buffer = numpy.array(potential_array)

    # Determine number of periodic images to scan and build image index list
    cdef double minimum_distance = min([abs(numpy.dot(plane_normal, vectors[index]) / numpy.linalg.norm(plane_normal)) \
        for index, plane_normal in enumerate(cell.normals)])
    cdef long max_images = numpy.ceil(distance / minimum_distance)
    cdef numpy.ndarray[numpy.int64_t, ndim=2] image_indices
    while True:
        try:
            image_indices = numpy.rollaxis(numpy.indices([(2 * max_images) + 1] * dimensions) \
                - max_images, 0, dimensions + 1).reshape(-1, dimensions)
            break
        except MemoryError:
            warnings.warn("insufficient memory to complete calculation; truncating with possible accuracy loss", RuntimeWarning)
            max_images /= 2
    cdef long all_zero_index = (image_indices.shape[0] - 1) / 2

    # Prepare histograms if requested
    cdef long histogram_spaces, histogram_space_left, histogram_space_right
    cdef double histogram_space, histogram_space_fraction
    cdef numpy.ndarray[numpy.float64_t, ndim=3] histograms
    if bin_width:
        histogram_spaces = numpy.ceil(distance / bin_width)
        histograms = numpy.zeros((atom_types, atom_types, histogram_spaces), dtype=numpy.float64)

    # Prepare for high performance calculations
    cdef double energy = 0.0
    cdef numpy.ndarray[numpy.float64_t, ndim=2] jacobian = numpy.zeros_like(atoms, dtype=numpy.float64)
    cdef long image_index, dimension_index_1, dimension_index_2, source_type_index, target_type_index, source_atom_index, target_atom_index
    cdef numpy.ndarray[numpy.float64_t, ndim=1] shift_vector = numpy.zeros(dimensions, dtype=numpy.float64)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] separation_vector = numpy.zeros(dimensions, dtype=numpy.float64)
    cdef double separation_distance, u, f, fi
    cdef Potential current_potential
    cdef long source_atom_identifier, target_atom_identifier

    # For each pair of atom types:
    for source_type_index in range(atom_types):
        for target_type_index in range(atom_types):

            # Retrieve the current potential
            current_potential = potential_buffer[source_type_index, target_type_index]

            # For each periodic image to consider:
            for image_index in range(image_indices.shape[0]):
                # Calculate the shift vector for the image
                for dimension_index_1 in range(dimensions):
                    shift_vector[dimension_index_1] = 0.0
                    for dimension_index_2 in range(dimensions):
                        shift_vector[dimension_index_1] += \
                            image_indices[image_index, dimension_index_2] * vectors[dimension_index_2, dimension_index_1]

                # For each pair of atoms satisfying the selected types:
                for source_atom_index in range(atom_counts[source_type_index]):
                    source_atom_identifier = atom_type_lookup[source_type_index] + source_atom_index
                    for target_atom_index in range(atom_counts[target_type_index]):
                        target_atom_identifier = atom_type_lookup[target_type_index] + target_atom_index

                        # Skip atoms which are in identical locations
                        if source_atom_identifier == target_atom_identifier \
                            and image_index == all_zero_index:
                            continue

                        # Newton's 3rd law optimization
                        if source_atom_identifier > target_atom_identifier:
                            continue

                        # Calculate the separation vector between the pair and the pair distance
                        separation_distance = 0.0
                        for dimension_index_1 in range(dimensions):
                            separation_vector[dimension_index_1] = shift_vector[dimension_index_1] + \
                                atoms[target_atom_identifier, dimension_index_1] - \
                                atoms[source_atom_identifier, dimension_index_1]
                            separation_distance += separation_vector[dimension_index_1] * separation_vector[dimension_index_1]
                        if separation_distance > distance * distance:
                            continue
                        separation_distance = libc.math.sqrt(separation_distance)

                        # Accumulate the energy
                        u, f = current_potential.evaluate(separation_distance)
                        if source_atom_identifier == target_atom_identifier:
                            # Periodic image self-interaction; reverse case counted also
                            energy += 0.5 * u
                        else:
                            # Newton's 3rd law optimization will skip reverse case
                            energy += u

                        # Accumulate the force (except for periodic image self-interactions)
                        if source_atom_identifier != target_atom_identifier:
                            for dimension_index_1 in range(dimensions):
                                fi = separation_vector[dimension_index_1] * f / separation_distance
                                jacobian[source_atom_identifier, dimension_index_1] += fi
                                jacobian[target_atom_identifier, dimension_index_1] -= fi

                        # Accumulate the histogram
                        if bin_width:
                            # Calculate which two bins to put the contact into
                            histogram_space = separation_distance / bin_width
                            histogram_space_right = libc.math.lround(histogram_space)
                            histogram_space_left = histogram_space_right - 1
                            histogram_space_fraction = 0.5 + histogram_space - histogram_space_right

                            # Handle edge cases at extreme ends of range
                            if histogram_space_left < 0:
                                histogram_space_left = 0
                            if histogram_space_right >= histogram_spaces:
                                histogram_space_right = histogram_spaces - 1

                            # Newton's 3rd law optimization based on atom identifiers, not indices
                            histograms[source_type_index, target_type_index, histogram_space_left] += 1.0 - histogram_space_fraction
                            histograms[source_type_index, target_type_index, histogram_space_right] += histogram_space_fraction
                            if source_atom_identifier != target_atom_identifier:
                                histograms[target_type_index, source_type_index, histogram_space_left] += 1.0 - histogram_space_fraction
                                histograms[target_type_index, source_type_index, histogram_space_right] += histogram_space_fraction

    # Return the final result, with histograms if they were requested
    energy_result = (energy / total_atom_count, jacobian.flatten() / total_atom_count)
    if bin_width:
        return (energy_result[0], energy_result[1], histograms)
    else:
        return energy_result
