"""Тестовые расчёты: сравнение численных результатов с аналитическими."""

import unittest

import numpy as np

from two_body import TwoBodyScatteringExperiment, Variant
from two_body.cross_section import DifferentialCrossSection
from two_body.kinematics import BinaryCollision
from two_body.particles import ParticleCatalog
from two_body.potentials import E2, CoulombPotential
from two_body.scattering import (
    ClosestApproachSolver,
    GaussMehlerScatteringAngle,
    RadialFunction,
)
from two_body.screening import (
    A_BOHR,
    BOHR,
    KR_C,
    MOLIERE,
    ZBL,
    FirsovScreeningLength,
    LindhardScharffScreeningLength,
)


class CoulombScatteringTest(unittest.TestCase):
    """Для кулоновского потенциала известны точные решения (резерфордовское рассеяние)."""

    def setUp(self):
        self.er = 100.0
        self.potential = CoulombPotential(1, 29)
        self.b = self.potential.z1 * self.potential.z2 * E2 / self.er
        self.g = RadialFunction(self.potential, self.er)
        self.solver = ClosestApproachSolver(self.g)
        self.rho = np.linspace(0.01, 2.0, 50) * self.b

    def test_closest_approach(self):
        expected = self.b / 2 + np.sqrt(self.b**2 / 4 + self.rho**2)
        np.testing.assert_allclose(self.solver.solve_many(self.rho), expected, rtol=1e-10)

    def test_gauss_mehler_angle(self):
        angle = GaussMehlerScatteringAngle(self.g, n_nodes=100)
        chi = angle.compute_many(self.rho, self.solver.solve_many(self.rho))
        expected = 2 * np.arctan(self.b / (2 * self.rho))
        np.testing.assert_allclose(chi, expected, atol=1e-4)

    def test_rutherford_cross_section(self):
        chi = 2 * np.arctan(self.b / (2 * self.rho))
        omega = DifferentialCrossSection(self.rho, chi)
        expected = (self.b / 4) ** 2 / np.sin(omega.theta / 2) ** 4
        np.testing.assert_allclose(omega.omega[1:-1], expected[1:-1], rtol=1e-2)


class KinematicsTest(unittest.TestCase):
    def setUp(self):
        self.collision = BinaryCollision(
            ParticleCatalog.get("p"), ParticleCatalog.get("Cu"), 0.1
        )

    def test_kinematic_factor_limits(self):
        m1, m2 = self.collision.m1, self.collision.m2
        self.assertAlmostEqual(float(self.collision.kinematic_factor(0.0)), 1.0)
        self.assertAlmostEqual(
            float(self.collision.kinematic_factor(np.pi)), ((m2 - m1) / (m1 + m2)) ** 2
        )

    def test_energy_conservation(self):
        chi = np.linspace(0.01, np.pi - 0.01, 50)
        theta1, theta2 = self.collision.lab_angles(chi)
        m1, m2 = self.collision.m1, self.collision.m2
        k1 = self.collision.kinematic_factor(theta1)
        k2 = 4 * m1 * m2 / (m1 + m2) ** 2 * np.cos(theta2) ** 2
        np.testing.assert_allclose(k1 + k2, 1.0, rtol=1e-12)


class ScreeningTest(unittest.TestCase):
    def test_coefficients_normalized(self):
        for function in (BOHR, MOLIERE, KR_C, ZBL):
            self.assertAlmostEqual(float(function(0.0)), 1.0, places=5)

    def test_screening_lengths(self):
        z1, z2 = 1, 29
        firsov = 0.8853 * A_BOHR * (z1**0.5 + z2**0.5) ** (-2 / 3)
        lindhard = 0.8853 * A_BOHR * (z1 ** (2 / 3) + z2 ** (2 / 3)) ** (-1 / 2)
        np.testing.assert_allclose(FirsovScreeningLength()(z1, z2), firsov, rtol=1e-4)
        np.testing.assert_allclose(LindhardScharffScreeningLength()(z1, z2), lindhard, rtol=1e-4)


class Variant4Test(unittest.TestCase):
    def test_variant_parameters(self):
        variant = Variant.from_number(4)
        potential = variant.build_potential()
        collision = variant.build_collision()
        self.assertIs(potential.screening_function, KR_C)
        self.assertIsInstance(potential.screening_length, FirsovScreeningLength)
        self.assertEqual((collision.projectile.symbol, collision.target.symbol), ("p", "Cu"))
        self.assertEqual(collision.energy, 0.1)

    def test_experiment(self):
        results = TwoBodyScatteringExperiment(Variant.from_number(4)).run()
        self.assertEqual(len(results.rho), 200)
        self.assertLess(abs(1 - results.closest_approach[-1] / results.rho_max), 1e-3)
        self.assertTrue(np.all(np.diff(results.chi) < 0))
        self.assertTrue(np.all((results.chi > 0) & (results.chi < np.pi)))
        self.assertTrue(np.all(results.cross_section1.omega > 0))


if __name__ == "__main__":
    unittest.main()
