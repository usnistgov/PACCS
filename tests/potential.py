#!/usr/bin/env python

import numpy
import scipy.optimize
import unittest

from paccs import crystal
from paccs import potential

class PotentialTests(unittest.TestCase):

    def test_evaluate(self):
        # This is the zero potential
        r = numpy.linspace(0.5, 2, 100)
        u, f = potential.Potential().evaluate_array(r)
        self.assertTrue(numpy.all(u == 0))
        self.assertTrue(numpy.all(f == 0))

    def test_evaluate_range(self):
        # Range evaluation simply works with 1-D numpy arrays
        potentials = [
            potential.Transform(potential.LennardJonesType()),
            potential.Piecewise(potential.Potential(), potential.Piecewise( \
                potential.JaglaType(), potential.Potential(), 1.05), 1)]
        r = numpy.linspace(0.5, 2, 100)
        for potential_object in potentials:
            u, f = potential_object.evaluate_array(r)
            for i in range(len(r)):
                ui, fi = potential_object.evaluate(r[i])
                self.assertEqual(u[i], ui)
                self.assertEqual(f[i], fi)

class TransformTests(unittest.TestCase):

    def test_force(self):
        # Check derivatives with finite difference approximation
        potential_object = potential.Transform(potential.LennardJonesType(), \
            1.23, 4.56, 0.078, 9.0)
        r = numpy.linspace(1.3, 2, 1000)
        u, f = potential_object.evaluate_array(r)
        f_approx = -numpy.gradient(u, r[1] - r[0])
        self.assertTrue(numpy.all(numpy.isclose(f[1:-1], f_approx[1:-1], rtol=1e-3, atol=1e-3)))

    def test_sigma(self):
        # Check length scaling and shifting
        potential1 = potential.LennardJonesType()
        potential2 = potential.Transform(potential1, sigma=1.23, s=0.0456)
        r1 = numpy.linspace(0.9, 2, 100)
        r2 = (r1 + 0.0456) * 1.23
        u1, f1 = potential1.evaluate_array(r1)
        u2, f2 = potential2.evaluate_array(r2)
        self.assertTrue(numpy.all(numpy.isclose(u1, u2)))
        self.assertTrue(numpy.all(numpy.isclose(f1, f2 * 1.23)))

    def test_epsilon(self):
        # Check energy scaling and shifting
        potential1 = potential.LennardJonesType()
        potential2 = potential.Transform(potential1, epsilon=1.23, phi=4.56)
        r = numpy.linspace(0.9, 2, 100)
        u1, f1 = potential1.evaluate_array(r)
        u2, f2 = potential2.evaluate_array(r)
        self.assertTrue(numpy.all(numpy.isclose(u1, (u2 / 1.23) - 4.56)))
        self.assertTrue(numpy.all(numpy.isclose(f1, f2 / 1.23)))

class PiecewiseTests(unittest.TestCase):

    def test_piecewise(self):
        # Check piecewise behavior (evaluating at cutoff calls far potential)
        potential1 = potential.JaglaType()
        potential2 = potential.LennardJonesType()
        piecewise = potential.Piecewise(potential1, potential2, 1.05)
        r1, r2 = numpy.linspace(1, 1.05, 100), numpy.linspace(1.05, 2, 100)
        ui1, fi1 = potential1.evaluate_array(r1)
        ui2, fi2 = potential2.evaluate_array(r2)
        up1, fp1 = piecewise.evaluate_array(r1)
        up2, fp2 = piecewise.evaluate_array(r2)
        self.assertTrue(numpy.all(numpy.isclose(ui1[:-1], up1[:-1])))
        self.assertTrue(numpy.all(numpy.isclose(ui2, up2)))
        self.assertTrue(numpy.all(numpy.isclose(fi1[:-1], fp1[:-1])))
        self.assertTrue(numpy.all(numpy.isclose(fi2, fp2)))

class DNACCTests(unittest.TestCase):

    def setUp(self):
        try:
            import dnacc
            from dnacc.units import nm
        except:
            print("*** cannot locate DNACC library, skipping associated unittest ***")
            self.__dnacc = False
        else:
            self.__dnacc = True

        self.__params = {"r1":100,
                "r2":100,
                "lengths":{"A":25, "B":50},
                "sigma1":{"A":1/25.**2},
                "sigma2":{"B":1/50.**2},
                "beta_DeltaG0":{("A","A"):-1, ("A","B"):-5, ("B","B"):0}
                }

    def test_bad_init_r1(self):
        if (self.__dnacc):
            self.__params["r1"] = -1
            with self.assertRaises(ValueError) as result:
                potential.DNACC(**self.__params)

    def test_bad_init_r2(self):
        if (self.__dnacc):
            self.__params["r2"] = -1
            with self.assertRaises(ValueError) as result:
                potential.DNACC(**self.__params)

    def test_bad_init_lengths(self):
        if (self.__dnacc):
            self.__params["lengths"]["A"] = -1
            with self.assertRaises(ValueError) as result:
                potential.DNACC(**self.__params)

    def test_bad_init_sigma1(self):
        if (self.__dnacc):
            self.__params["sigma1"]["A"] = -1
            with self.assertRaises(ValueError) as result:
                potential.DNACC(**self.__params)

    def test_bad_init_sigma1_construct(self):
        if (self.__dnacc):
            self.__params["sigma1"]["C"] = 1.0
            with self.assertRaises(Exception) as result:
                potential.DNACC(**self.__params)

    def test_bad_init_sigma2(self):
        if (self.__dnacc):
            self.__params["sigma2"]["A"] = -1
            with self.assertRaises(ValueError) as result:
                potential.DNACC(**self.__params)

    def test_bad_init_sigma2_construct(self):
        if (self.__dnacc):
            self.__params["sigma2"]["C"] = 1.0
            with self.assertRaises(Exception) as result:
                potential.DNACC(**self.__params)

    def test_bad_init_beta_DeltaG0_symm(self):
        if (self.__dnacc):
            self.__params["beta_DeltaG0"][("B", "A")] = -4.0
            with self.assertRaises(ValueError) as result:
                potential.DNACC(**self.__params)

    def test_bad_init_beta_DeltaG0_miss(self):
        if (self.__dnacc):
            del self.__params["beta_DeltaG0"][("A", "A")]
            with self.assertRaises(Exception) as result:
                potential.DNACC(**self.__params)

    def test_bad_init_beta_DeltaG0_excess(self):
        if (self.__dnacc):
            self.__params["beta_DeltaG0"][("A", "C")] = -1.0
            with self.assertRaises(Exception) as result:
                potential.DNACC(**self.__params)

    def test_bad_init_beta(self):
        if (self.__dnacc):
            self.__params["beta"] = -1.0
            with self.assertRaises(ValueError) as result:
                potential.DNACC(**self.__params)

    def test_good_init(self):
        if (self.__dnacc):
            fail = False
            try:
                potential.DNACC(**self.__params)
            except:
                fail = True
            self.assertFalse(fail)

    def test_reduce(self):
        if (self.__dnacc):
            res = potential.DNACC(**self.__params)
            red = res.__reduce__()
            x = dict(zip(res.__pnames__(), red[1]))
            red[0](**x)

            fail = False
            try:
                red[0](**x)
            except:
                fail = True
            self.assertFalse(fail)

    def test_potential_1(self):
        if (self.__dnacc):
            try:
                import dnacc
                from dnacc.units import nm
            except:
                pass
            else:
                self.__params = {"r1":500,
                        "r2":500,
                        "lengths":{"A":20, "B":20},
                        "sigma1":{"A":1/20.**2},
                        "sigma2":{"B":1/20.**2},
                        "beta_DeltaG0":{("A","A"):0, ("A","B"):-8, ("B","B"):0}
                        }
                res = potential.DNACC(**self.__params)
                r = numpy.linspace(41/1000., 41, 1000)+500.+500.
                u,f = res.evaluate_array(r)

                # Example from documentation
                plates = dnacc.PlatesMeanField()
                plates.add_tether_type(plate='lower',
                                       sticky_end='alpha',
                                       L=20 * nm,
                                       sigma=1 / (20 * nm) ** 2)
                plates.add_tether_type(plate='upper',
                                       sticky_end='alphap',
                                       L=20 * nm,
                                       sigma=1 / (20 * nm) ** 2)
                plates.beta_DeltaG0['alpha', 'alphap'] = -8
                plates.at(41 * nm).set_reference_now()
                V_plate_arr = [plates.at(rv-1000.).free_energy_density for rv in r]
                R = 500 * nm
                V_sphere_arr = dnacc.calc_spheres_potential(r-1000., V_plate_arr, R)

                # Compare results over the domain of the theory
                self.assertTrue(numpy.allclose(V_sphere_arr, u))

                # Check that potential is zero beyond cutoff
                self.assertEqual(res.evaluate(r[-1]+0.00001), (0.0, 0.0))

                # Check that WCA-esque potential wall is used at contact and below
                rep_wallu, rep_wallf = res.evaluate_array(numpy.linspace(0, r[0], 1000))

                # Wall is monotonically decreasing
                self.assertEqual(res.evaluate(0), (numpy.inf, numpy.inf))
                self.assertTrue(numpy.all(rep_wallu[1:] < numpy.inf) and numpy.all(rep_wallu[1:] > 0))
                self.assertTrue(numpy.all(x>y for x, y in zip(rep_wallu, rep_wallu[1:])))
                self.assertTrue(numpy.all(rep_wallf[1:] < numpy.inf) and numpy.all(rep_wallf[1:] > 0))
                self.assertTrue(numpy.all(x>y for x, y in zip(rep_wallf, rep_wallf[1:])))

                # Wall connects continuously
                self.assertTrue(numpy.isclose(rep_wallu[-1], u[0]))
                self.assertTrue(numpy.isclose(rep_wallf[-1], f[0]))

class LennardJonesTypeTests(unittest.TestCase):

    def test_force(self):
        # Check derivatives with finite difference approximation
        potential_object = potential.LennardJonesType(1.2, 3.4, 0.5, 6.7, 0.89)
        r = numpy.linspace(0.75, 2, 1000)
        u, f = potential_object.evaluate_array(r)
        f_approx = -numpy.gradient(u, r[1] - r[0])
        self.assertTrue(numpy.all(numpy.isclose(f[1:-1], f_approx[1:-1], rtol=1e-3, atol=1e-3)))

    def test_attractive(self):
        # Check attractive potential for well
        potential_object = potential.LennardJonesType(s=1)
        r1, r2 = numpy.linspace(0.5, 1, 100)[:-1], numpy.linspace(1, 1.5, 100)[1:]
        f1, f2 = potential_object.evaluate_array(r1)[1], potential_object.evaluate_array(r2)[1]
        self.assertTrue(numpy.all(f1 > 0))
        self.assertTrue(numpy.all(f2 < 0))

    def test_repulsive(self):
        # Check repulsive potential for lack of well
        potential_object = potential.LennardJonesType(s=1, lambda_=-1)
        self.assertTrue(numpy.all(potential_object.evaluate_array(numpy.linspace(0.5, 1.5, 100))[1] > 0))

    def test_sigma(self):
        # Check length scaling
        reference_r = numpy.linspace(0.75, 2, 100)
        reference_u, reference_f = potential.LennardJonesType().evaluate_array(reference_r)
        for sigma in numpy.linspace(0.5, 3, 10):
            test_u, test_f = potential.LennardJonesType(sigma=sigma).evaluate_array(reference_r * sigma)
            self.assertTrue(numpy.all(numpy.isclose(reference_u, test_u)))
            self.assertTrue(numpy.all(numpy.isclose(reference_f, test_f * sigma)))

    def test_epsilon(self):
        # Check energy scaling
        reference_r = numpy.linspace(0.75, 2, 100)
        reference_u, reference_f = potential.LennardJonesType().evaluate_array(reference_r)
        for epsilon in numpy.linspace(0.5, 3, 10):
            test_u, test_f = potential.LennardJonesType(epsilon=epsilon).evaluate_array(reference_r)
            self.assertTrue(numpy.all(numpy.isclose(reference_u, test_u / epsilon)))
            self.assertTrue(numpy.all(numpy.isclose(reference_f, test_f / epsilon)))

    def test_lambda(self):
        # Check shape factor
        reference_r = numpy.linspace(0.75, 2, 100)
        last_u, last_f = potential.LennardJonesType().evaluate_array(reference_r)
        for lambda_ in numpy.linspace(-1, 1, 10)[::-1][1:]:
            test_u, test_f = potential.LennardJonesType(lambda_=lambda_).evaluate_array(reference_r)
            self.assertTrue(numpy.all(test_u > last_u))
            self.assertTrue(numpy.all(test_f > last_f))
            last_u, last_f = test_u, test_f

    def test_n(self):
        # Check potential sharpness changing
        reference_r = numpy.linspace(0.75, 2, 100)
        last_u = potential.LennardJonesType().evaluate_array(reference_r)[0]
        for n in numpy.linspace(6, 96, 10)[1:]:
            test_u = potential.LennardJonesType(n=n).evaluate_array(reference_r)[0]
            self.assertTrue(numpy.all(test_u >= last_u))
            last_u = test_u

    def test_s(self):
        # Check horizontal shift
        reference_r = numpy.linspace(0.75, 2, 100)
        reference_u, reference_f = potential.LennardJonesType(s=1.0).evaluate_array(reference_r)
        for s in numpy.linspace(0.5, 2.5, 10):
            test_u, test_f = potential.LennardJonesType(s=s).evaluate_array(reference_r + s - 1.0)
            self.assertTrue(numpy.all(numpy.isclose(reference_u, test_u)))
            self.assertTrue(numpy.all(numpy.isclose(reference_f, test_f)))

class JaglaTypeTests(unittest.TestCase):

    def test_force(self):
        # Check derivatives with finite difference approximation
        potential_object = potential.JaglaType()
        r = numpy.linspace(1.0, 1.05, 10000)
        u, f = potential_object.evaluate_array(r)
        f_approx = -numpy.gradient(u, r[1] - r[0])
        self.assertTrue(numpy.all(numpy.isclose(f[1:-1], f_approx[1:-1], rtol=1e-3, atol=1e-3)))

    def test_make(self):
        # Make sure solver can find minimum properly
        for energy in numpy.linspace(0, 1, 10):
            make_result = potential.JaglaType.make(energy)
            potential_object = potential.JaglaType(*make_result[0])
            minimum = scipy.optimize.minimize_scalar(lambda r: potential_object.evaluate(r)[0], \
                method="bounded", bounds=(1.0, 1.05), options={"xatol": 0.0})
            self.assertAlmostEqual(-energy, minimum.fun)
            self.assertAlmostEqual(make_result[1], minimum.x)

class EvaluateTests(unittest.TestCase):

    # Try many different cells
    _cases = [
        (crystal.Cell(2.1 * numpy.eye(3), [numpy.array([[0.01, 0.02, 0.03]]), numpy.array([[0.98, 0.97, 0.99]])]),
        {(0, 0): potential.LennardJonesType(s=3.0**0.5, lambda_=-1),
        (1, 1): potential.LennardJonesType(s=3.0**0.5, lambda_=-1),
        (0, 1): potential.LennardJonesType(s=3.0**0.5)}, 6, None),

        (crystal.Cell(100 * numpy.eye(3), [numpy.array([[49, 50, 50], [51, 50, 50]])]),
        {(0, 0): potential.LennardJonesType(s=2)}, 3, -0.5),

        (crystal.Cell(100 * numpy.eye(3), [numpy.array([[49, 50, 50], [51, 50, 50]])]),
        {(0, 0): potential.LennardJonesType()}, 3, None),

        (crystal.Cell(100 * numpy.eye(3), [numpy.array([[99, 50, 50], [1, 50, 50]])]),
        {(0, 0): potential.LennardJonesType(s=2)}, 3, -0.5),

        (crystal.Cell(100 * numpy.eye(3), [numpy.array([[99, 50, 50], [1, 50, 50]])]),
        {(0, 0): potential.LennardJonesType()}, 3, None),

        (crystal.Cell(100 * numpy.eye(3), [numpy.array([[49, 49, 50], [51, 51, 50], [49, 51, 50], [51, 49, 50]])]),
        {(0, 0): potential.LennardJonesType()}, 3, None),

        (crystal.Cell(100 * numpy.eye(3), [numpy.array([[-1, -1, 50], [1, 1, 50], [-1, 1, 50], [1, -1, 50]])]),
        {(0, 0): potential.LennardJonesType()}, 3, None),

        (crystal.Cell(numpy.array([[100, 0, 0], [100, 100, 0], [0, 0, 100]]),
        [numpy.array([[49, 49, 50], [51, 51, 50], [51, 49, 50], [149, 51, 50]])]),
        {(0, 0): potential.LennardJonesType()}, 3, None),

        (crystal.Cell(numpy.array([[100, 0, 0], [100, 100, 0], [0, 0, 100]]),
        [numpy.array([[49, 49, 50], [51, 51, 50]]), numpy.array([[51, 49, 50], [149, 51, 50]])]),
        {(0, 0): potential.LennardJonesType(), (0, 1): potential.LennardJonesType(),
        (1, 1): potential.LennardJonesType(lambda_=-1)}, 3, None),

        (crystal.Cell(numpy.array([[100, 0, 0], [100, 100, 0], [100, 100, 100]]),
        [numpy.array([[100, 100, 100], [101, 101, 101], [102, 102, 102], [103, 103, 103]])]),
        {(0, 0): potential.LennardJonesType()}, 3, None),

        (crystal.Cell(numpy.eye(3), [0.5 * numpy.zeros((1, 3))]), {(0, 0): potential.LennardJonesType(s=1)}, 6, None),

        (crystal.Cell(numpy.array([[1, 0, 0], [6, 1, 0], [0, 0, 1]]), [numpy.array([[3, 0.5, 0.5]])]),
        {(0, 0): potential.LennardJonesType(s=1)}, 6, None),

        (crystal.Cell(numpy.array([[1, 0.01, 0.02], [0.03, 1, 0.04], [0.05, 0.06, 1]]), [numpy.array([[0.25, 0.26, 0.27]]),
        numpy.array([[0.75, 0.76, 0.77]])]), {(0, 0): potential.LennardJonesType(s=0.7, n=24), (1, 1): potential.LennardJonesType(s=0.8, n=12),
        (0, 1): potential.LennardJonesType(s=0.75, n=9)}, 6, None),

        (crystal.CellTools.wrap(crystal.Cell(numpy.array([[1, 0.01, 0.02], [0.03, 1, 0.04], [0.05, 0.06, 1]]),
        [numpy.array([[0.95, 0.96, 0.97]]), numpy.array([[1.45, 1.46, 1.47]])])),
        {(0, 0): potential.LennardJonesType(s=0.7, n=24), (1, 1): potential.LennardJonesType(s=0.8, n=12),
        (0, 1): potential.LennardJonesType(s=0.75, n=9)}, 6, None),

        (crystal.Cell(numpy.array([[1.1, 0.2, 0.4], [0.6, 1, 0.8], [1.2, 1.4, 1]]),
        [numpy.array([[1, 1, 1.1], [1.6, 1.8, 1.7]])]), {(0, 0): potential.LennardJonesType(s=0.3, n=24)}, 6, None),

        (crystal.Cell(numpy.array([[2.66703688, -0.32850482, -0.12184191], [1.11351198, 1.61363475, 0.0948566],
        [0.98523977, 0.40819993, 1.64547459]]), [numpy.array([[0.05395002, -0.09955472, -0.12233975]]),
        numpy.array([[0.76254656, 0.57095924, 0.45567309]])]), {(0, 0): potential.LennardJonesType(lambda_=-1),
        (1, 1): potential.LennardJonesType(lambda_=-1), (0, 1): potential.LennardJonesType()}, 6, -1.33496937438),

        (crystal.Cell(numpy.array([[2.6899529, 0, 0], [0.90266756, 1.74296159, 0],
        [0.85246376, 0.65541009, 1.63971167]]), [numpy.array([[0.07118973, -0.1012279, -0.1116857]]),
        numpy.array([[0.66568335, 0.69580262, 0.43339827]])]), {(0, 0): potential.LennardJonesType(lambda_=-1),
        (1, 1): potential.LennardJonesType(lambda_=-1), (0, 1): potential.LennardJonesType()}, 6, -1.33496937438),
    ]

    def test_methods_jacobian(self):
        for cell, potentials, cutoff, expected in EvaluateTests._cases:
            # First, make sure Python and Cython algorithms give identical answers
            cell = crystal.CellTools.reduce(cell)
            energy_slow = crystal.CellTools.energy(cell, potentials, cutoff)
            energy_fast, jacobian_fast = potential._evaluate_fast(cell, potentials, cutoff)
            self.assertAlmostEqual(energy_slow, energy_fast)
            if expected is not None:
                self.assertTrue(numpy.isclose(energy_fast, expected))

            # Now use second-order finite differences to make sure gradient is correct
            delta = 2.0 ** -26.0
            index = 0
            for type_index in range(cell.atom_types):
                for atom_index in range(cell.atom_count(type_index)):
                    for component_index in range(cell.dimensions):
                        low_lists, high_lists = cell.atom_lists, cell.atom_lists
                        low_lists[type_index][atom_index][component_index] -= delta
                        high_lists[type_index][atom_index][component_index] += delta
                        low_energy = potential._evaluate_fast(crystal.CellTools.wrap(crystal.Cell(cell.vectors, low_lists)), potentials, cutoff)[0]
                        high_energy = potential._evaluate_fast(crystal.CellTools.wrap(crystal.Cell(cell.vectors, high_lists)), potentials, cutoff)[0]
                        derivative = (high_energy - low_energy) / (delta * 2)
                        self.assertTrue(numpy.isclose(derivative, jacobian_fast[index], atol=1e-4, rtol=1e-4))
                        index += 1

    def test_histogram_correct(self):
        # Construct a histogram manually using cell RDF method
        cell = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), numpy.array([[0.3] * 3, [0.65] * 3])])
        histogram = numpy.zeros((2, 2, 20))
        for source in range(2):
            for target in range(2):
                for key, value in cell.rdf(source, target, 5).items():
                    factor = 0.5 + (4 * key) - int(numpy.round(4 * key))
                    histogram[source, target, max(0, int(numpy.round(4 * key)) - 1)] += value * (1 - factor)
                    histogram[source, target, min(19, int(numpy.round(4 * key)))] += value * factor

        # Construct a fast histogram and compare
        self.assertTrue(numpy.all(numpy.isclose(histogram, potential._evaluate_fast(cell, None, 5, 0.25)[2])))

    def test_histogram_binning(self):
        # Check for proper binning (number of bins)
        cell = crystal.Cell(numpy.eye(3), [numpy.zeros((1, 3)), numpy.array([[0.3] * 3, [0.65] * 3])])
        self.assertEqual(potential._evaluate_fast(cell, None, 6, 0.25)[2].shape[2], 24)
        self.assertEqual(potential._evaluate_fast(cell, None, 6.01, 0.25)[2].shape[2], 25)
        self.assertEqual(potential._evaluate_fast(cell, None, 4, 5)[2].shape[2], 1)

    def test_double(self):
        # Simple check with pairwise interaction direction
        energy, jacobian = potential._evaluate_fast(*EvaluateTests._cases[2][:3])
        self.assertLess(jacobian[0], 0)
        self.assertGreater(jacobian[3], 0)

    def test_double_wrap(self):
        # Make sure wrapping is working properly
        energy, jacobian = potential._evaluate_fast(*EvaluateTests._cases[4][:3])
        self.assertLess(jacobian[0], 0)
        self.assertGreater(jacobian[3], 0)

    def test_skewed(self):
        # Make sure that highly distorted cells evaluate properly
        energy_1, jacobian_1 = potential._evaluate_fast(*EvaluateTests._cases[10][:3])
        energy_2, jacobian_2 = potential._evaluate_fast(*EvaluateTests._cases[11][:3])
        self.assertTrue(numpy.all(numpy.isclose(energy_1, energy_2)))
        self.assertTrue(numpy.all(numpy.isclose(jacobian_1, jacobian_2)))

    def test_translated(self):
        # Make sure that an arbitrary translation does not affect anything
        energy_1, jacobian_1 = potential._evaluate_fast(*EvaluateTests._cases[12][:3])
        energy_2, jacobian_2 = potential._evaluate_fast(*EvaluateTests._cases[13][:3])
        self.assertTrue(numpy.all(numpy.isclose(energy_1, energy_2)))
        self.assertTrue(numpy.all(numpy.isclose(jacobian_1, jacobian_2)))
