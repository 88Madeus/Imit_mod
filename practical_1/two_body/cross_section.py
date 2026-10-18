"""Дифференциальное сечение упругого рассеяния."""

import numpy as np


class DifferentialCrossSection:
    """Сечение (19): ω(θ) = ρ(θ) / sin θ * |dρ/dθ|, нм^2/ср.

    Зависимость ρ(θ) задаётся таблично (обратная к θ(ρ)), производная
    находится численно по неравномерной сетке.
    """

    def __init__(self, rho, theta):
        rho = np.asarray(rho, dtype=float)
        theta = np.asarray(theta, dtype=float)
        order = np.argsort(theta)
        self.theta = theta[order]
        self.rho = rho[order]
        self.omega = self.rho / np.sin(self.theta) * np.abs(np.gradient(self.rho, self.theta))
