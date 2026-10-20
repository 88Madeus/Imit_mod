"""Потенциалы взаимодействия двух частиц."""

from abc import ABC, abstractmethod

import numpy as np

from .screening import ScreeningFunction, ScreeningLength

E2 = 1.44  # квадрат элементарного заряда e^2, эВ*нм


class InteractionPotential(ABC):
    """Потенциальная энергия взаимодействия U(r), эВ (r в нм)."""

    def __init__(self, z1: int, z2: int):
        self.z1 = z1
        self.z2 = z2

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __call__(self, r):
        ...


class CoulombPotential(InteractionPotential):
    """Кулоновский потенциал: U(r) = Z1 Z2 e^2 / r (используется в тестах)."""

    @property
    def name(self) -> str:
        return "Кулоновский"

    def __call__(self, r):
        return self.z1 * self.z2 * E2 / np.asarray(r, dtype=float)


class ScreenedCoulombPotential(CoulombPotential):
    """Экранированный кулоновский потенциал: U(r) = Z1 Z2 e^2 / r * F(r / a)."""

    def __init__(
        self,
        z1: int,
        z2: int,
        screening_function: ScreeningFunction,
        screening_length: ScreeningLength,
    ):
        super().__init__(z1, z2)
        self.screening_function = screening_function
        self.screening_length = screening_length
        self.a = screening_length(z1, z2)

    @property
    def name(self) -> str:
        return (
            f"{self.screening_function.name} "
            f"(радиус экранирования {self.screening_length.name})"
        )

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        return super().__call__(r) * self.screening_function(r / self.a)
