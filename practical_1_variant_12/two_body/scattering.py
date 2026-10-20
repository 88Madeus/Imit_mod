"""Расчёт расстояния наименьшего сближения, сетки прицельных параметров
и угла рассеяния в системе центра масс."""

import numpy as np

from .potentials import InteractionPotential


class RadialFunction:
    """g(r) = 1 - U(r) / Er - ρ^2 / r^2 — подкоренное выражение интеграла рассеяния (16)."""

    def __init__(self, potential: InteractionPotential, relative_energy: float):
        self.potential = potential
        self.relative_energy = relative_energy

    def __call__(self, r, rho: float):
        r = np.asarray(r, dtype=float)
        return 1.0 - self.potential(r) / self.relative_energy - (rho / r) ** 2


class ClosestApproachSolver:
    """Решает уравнение g(R) = 0 относительно расстояния наименьшего сближения R
    методом деления отрезка пополам (для отталкивающего потенциала g(r) монотонно
    возрастает, поэтому корень единственный)."""

    def __init__(self, g: RadialFunction, rel_tol: float = 1e-14, max_iter: int = 200):
        self.g = g
        self.rel_tol = rel_tol
        self.max_iter = max_iter

    def __call__(self, rho: float) -> float:
        hi = max(rho, 1e-3)
        while self.g(hi, rho) < 0:
            hi *= 2.0
        lo = hi
        while self.g(lo, rho) >= 0:
            lo /= 2.0
        for _ in range(self.max_iter):
            mid = 0.5 * (lo + hi)
            if self.g(mid, rho) < 0:
                lo = mid
            else:
                hi = mid
            if hi - lo <= self.rel_tol * hi:
                break
        # Возвращаем правую границу: на ней g(R) >= 0, что гарантирует
        # неотрицательность подкоренного выражения при r > R.
        return hi

    def solve_many(self, rho) -> np.ndarray:
        return np.array([self(value) for value in np.atleast_1d(rho)])


class ImpactParameterGrid:
    """Сетка прицельных параметров ρ в интервале [ρmax/n, ρmax] с шагом ρmax/n,
    где ρmax — наименьший ρ, для которого |1 - R/ρ| < tolerance."""

    def __init__(
        self,
        solver: ClosestApproachSolver,
        tolerance: float = 1e-3,
        n_points: int = 200,
    ):
        self.solver = solver
        self.tolerance = tolerance
        self.n_points = n_points

    def _deviation(self, rho: float) -> float:
        return abs(1.0 - self.solver(rho) / rho)

    def find_rho_max(self) -> float:
        hi = self.solver(0.0)  # расстояние сближения при лобовом ударе
        while self._deviation(hi) >= self.tolerance:
            hi *= 2.0
        lo = hi / 2.0
        if self._deviation(lo) < self.tolerance:
            return lo
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if self._deviation(mid) < self.tolerance:
                hi = mid
            else:
                lo = mid
            if hi - lo <= 1e-12 * hi:
                break
        return hi

    def build(self) -> np.ndarray:
        rho_max = self.find_rho_max()
        step = rho_max / self.n_points
        return step * np.arange(1, self.n_points + 1)


class GaussMehlerScatteringAngle:
    """Угол рассеяния в ц-системе χ(ρ) по квадратурной формуле Гаусса–Мелера:

        χ = π - 2 (ρ/R) (π/N) Σ ξ_n ξ_{N-n+1} / sqrt(g(R / ξ_n^2)),
        ξ_n = cos((2n - 1) π / (4N)).
    """

    def __init__(self, g: RadialFunction, n_nodes: int = 100):
        self.g = g
        self.n_nodes = n_nodes
        n = np.arange(1, n_nodes + 1)
        self.nodes = np.cos((2 * n - 1) * np.pi / (4 * n_nodes))
        self.weights = self.nodes * self.nodes[::-1]

    def __call__(self, rho: float, closest_approach: float) -> float:
        r = closest_approach / self.nodes**2
        integrand = self.weights / np.sqrt(self.g(r, rho))
        return np.pi - 2.0 * rho / closest_approach * np.pi / self.n_nodes * integrand.sum()

    def compute_many(self, rho, closest_approach) -> np.ndarray:
        return np.array([self(p, r) for p, r in zip(rho, closest_approach)])
