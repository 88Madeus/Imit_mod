"""Кинематика упругого столкновения налетающей частицы с покоящейся."""

import numpy as np

from .particles import Particle


class BinaryCollision:
    """Парное упругое столкновение: налетающая частица m1 с энергией E0 и покоящаяся m2."""

    def __init__(self, projectile: Particle, target: Particle, energy: float):
        if energy <= 0:
            raise ValueError("Энергия налетающей частицы должна быть положительной")
        self.projectile = projectile
        self.target = target
        self.energy = energy  # E0, эВ

    @property
    def m1(self) -> float:
        return self.projectile.mass

    @property
    def m2(self) -> float:
        return self.target.mass

    @property
    def relative_energy(self) -> float:
        """Энергия в системе центра масс: Er = E0 * m2 / (m1 + m2)."""
        return self.energy * self.m2 / (self.m1 + self.m2)

    @property
    def max_lab_angle(self) -> float:
        """Максимальный угол рассеяния налетающей частицы в л-системе."""
        if self.m1 <= self.m2:
            return np.pi
        return float(np.arcsin(self.m2 / self.m1))

    def kinematic_factor(self, theta1):
        """Кинематический фактор (15): k = E1 / E0 как функция угла θ1.

        Для m1 > m2 возвращается ветвь со знаком «+», за пределами
        максимального угла рассеяния — NaN.
        """
        theta1 = np.asarray(theta1, dtype=float)
        ratio = self.m2 / self.m1
        discriminant = ratio**2 - np.sin(theta1) ** 2
        root = np.sqrt(np.where(discriminant >= 0, discriminant, np.nan))
        return (self.m1 / (self.m1 + self.m2)) ** 2 * (np.cos(theta1) + root) ** 2

    def scattered_energy(self, theta1):
        """Энергия налетающей частицы после рассеяния E1 = k * E0, эВ."""
        return self.kinematic_factor(theta1) * self.energy

    def lab_angles(self, chi):
        """Углы рассеяния в л-системе (17) по углу χ в ц-системе: (θ1, θ2)."""
        chi = np.asarray(chi, dtype=float)
        theta1 = np.arctan2(self.m2 * np.sin(chi), self.m1 + self.m2 * np.cos(chi))
        theta2 = (np.pi - chi) / 2.0
        return theta1, theta2
