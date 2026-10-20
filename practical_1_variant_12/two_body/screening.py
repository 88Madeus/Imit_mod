"""Функция экранирования Kr-C и радиус экранирования Фирсова."""

from abc import ABC, abstractmethod

import numpy as np

A_BOHR = 0.0529  # первый боровский радиус, нм
THOMAS_FERMI_CONSTANT = (9.0 * np.pi**2 / 128.0) ** (1.0 / 3.0)  # 0,8853


class ScreeningFunction:
    """Функция экранирования F(x) = sum c_i * exp(-d_i * x), где x = r / a."""

    def __init__(self, name: str, c, d):
        self.name = name
        self.c = np.asarray(c, dtype=float)
        self.d = np.asarray(d, dtype=float)
        if self.c.shape != self.d.shape:
            raise ValueError("Количество коэффициентов c_i и d_i должно совпадать")
        if not np.isclose(self.c.sum(), 1.0, atol=1e-5):
            raise ValueError(f"Сумма c_i должна быть равна 1, получено {self.c.sum()}")

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        return np.sum(self.c * np.exp(-np.multiply.outer(x, self.d)), axis=-1)

    def __repr__(self) -> str:
        return f"ScreeningFunction({self.name!r})"


KR_C = ScreeningFunction(
    "Kr-C", [0.190945, 0.473674, 0.335381], [0.278544, 0.637174, 1.919249]
)


class ScreeningLength(ABC):
    """Длина экранирования a(Z1, Z2), нм."""

    name = ""

    @abstractmethod
    def __call__(self, z1: int, z2: int) -> float:
        ...


class FirsovScreeningLength(ScreeningLength):
    """Радиус экранирования Фирсова: a = 0,8853 a_B (Z1^1/2 + Z2^1/2)^(-2/3)."""

    name = "Фирсова"

    def __call__(self, z1: int, z2: int) -> float:
        return THOMAS_FERMI_CONSTANT * A_BOHR * (z1**0.5 + z2**0.5) ** (-2.0 / 3.0)
