"""Вычислительный эксперимент: последовательное выполнение пп. 1–4 задания."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cross_section import DifferentialCrossSection
from .kinematics import BinaryCollision
from .potentials import InteractionPotential
from .scattering import (
    ClosestApproachSolver,
    GaussMehlerScatteringAngle,
    ImpactParameterGrid,
    RadialFunction,
)
from .variant import Variant


@dataclass
class ScatteringResults:
    """Результаты расчёта для одного варианта."""

    variant: Variant
    collision: BinaryCollision
    potential: InteractionPotential
    # п. 1а: потенциальная энергия
    r: np.ndarray
    potential_energy: np.ndarray
    # п. 1б: кинематический фактор и энергия после рассеяния
    theta1_grid: np.ndarray
    kinematic_factor: np.ndarray
    scattered_energy: np.ndarray
    # п. 2: сетка прицельных параметров и расстояние наименьшего сближения
    rho_max: float
    rho: np.ndarray
    closest_approach: np.ndarray
    # п. 3: углы рассеяния
    chi: np.ndarray
    theta1: np.ndarray
    theta2: np.ndarray
    # п. 4: дифференциальные сечения
    cross_section1: DifferentialCrossSection
    cross_section2: DifferentialCrossSection

    def save_csv(self, path: Path) -> None:
        """Таблица ρ, R, χ, θ1, θ2 (углы в градусах) для отчёта."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rho_nm", "R_nm", "chi_deg", "theta1_deg", "theta2_deg"])
            for row in zip(
                self.rho,
                self.closest_approach,
                np.degrees(self.chi),
                np.degrees(self.theta1),
                np.degrees(self.theta2),
            ):
                writer.writerow([f"{value:.8g}" for value in row])


class TwoBodyScatteringExperiment:
    """Решение задачи двух тел для заданного варианта."""

    def __init__(
        self,
        variant: Variant,
        n_nodes: int = 100,
        n_rho: int = 200,
        rho_tolerance: float = 1e-3,
        n_angles: int = 361,
    ):
        self.variant = variant
        self.collision = variant.build_collision()
        self.potential = variant.build_potential()
        self.g = RadialFunction(self.potential, self.collision.relative_energy)
        self.solver = ClosestApproachSolver(self.g)
        self.grid = ImpactParameterGrid(self.solver, rho_tolerance, n_rho)
        self.angle = GaussMehlerScatteringAngle(self.g, n_nodes)
        self.n_angles = n_angles

    def run(self) -> ScatteringResults:
        # п. 2: сетка прицельных параметров и R(ρ)
        rho = self.grid.build()
        closest_approach = self.solver.solve_many(rho)

        # п. 1а: U(r) в интервале, охватывающем все расстояния сближения
        r = np.geomspace(closest_approach.min() / 10.0, rho[-1] * 2.0, 500)
        potential_energy = self.potential(r)

        # п. 1б: k(θ1) и E1(θ1)
        theta1_grid = np.linspace(0.0, self.collision.max_lab_angle, self.n_angles)
        kinematic_factor = self.collision.kinematic_factor(theta1_grid)
        scattered_energy = kinematic_factor * self.collision.energy

        # п. 3: χ(ρ), θ1(ρ), θ2(ρ)
        chi = self.angle.compute_many(rho, closest_approach)
        theta1, theta2 = self.collision.lab_angles(chi)

        # п. 4: ρ(θ) и ω(θ)
        cross_section1 = DifferentialCrossSection(rho, theta1)
        cross_section2 = DifferentialCrossSection(rho, theta2)

        return ScatteringResults(
            variant=self.variant,
            collision=self.collision,
            potential=self.potential,
            r=r,
            potential_energy=potential_energy,
            theta1_grid=theta1_grid,
            kinematic_factor=kinematic_factor,
            scattered_energy=scattered_energy,
            rho_max=float(rho[-1]),
            rho=rho,
            closest_approach=closest_approach,
            chi=chi,
            theta1=theta1,
            theta2=theta2,
            cross_section1=cross_section1,
            cross_section2=cross_section2,
        )
