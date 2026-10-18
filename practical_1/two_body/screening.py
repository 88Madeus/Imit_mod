"""Функции и длины экранирования для экранированных кулоновских потенциалов."""

from abc import ABC, abstractmethod

import numpy as np

A_BOHR = 0.0529  # первый боровский радиус, нм
THOMAS_FERMI_CONSTANT = (9.0 * np.pi**2 / 128.0) ** (1.0 / 3.0)  # 0,8853


class ScreeningFunction:
    """Функция экранирования F(x) = sum c_i * exp(-d_i * x), где x = r / a (табл. 1)."""

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


BOHR = ScreeningFunction("Бора", [1.0], [1.0])
MOLIERE = ScreeningFunction("Мольера", [0.35, 0.55, 0.10], [0.3, 1.2, 6.0])
KR_C = ScreeningFunction(
    "Kr-C", [0.190945, 0.473674, 0.335381], [0.278544, 0.637174, 1.919249]
)
# В табл. 1 пособия коэффициенты ZBL дают sum c_i = 0,945 (F(0) != 1),
# поэтому используются стандартные коэффициенты Циглера–Бирзака–Литтмарка.
ZBL = ScreeningFunction(
    "ZBL", [0.02817, 0.28022, 0.50986, 0.18175], [0.20162, 0.40290, 0.94229, 3.19980]
)


class ScreeningLength(ABC):
    """Длина экранирования a(Z1, Z2), нм."""

    name = ""

    @abstractmethod
    def __call__(self, z1: int, z2: int) -> float:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class ThomasFermiScreeningLength(ScreeningLength):
    """a = 0,8853 * a_B * Z12^(-1/3), где Z12 = (Z1^x + Z2^x)^y и x*y = 1."""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def effective_charge(self, z1: int, z2: int) -> float:
        return (z1**self.x + z2**self.x) ** self.y

    def __call__(self, z1: int, z2: int) -> float:
        return THOMAS_FERMI_CONSTANT * A_BOHR * self.effective_charge(z1, z2) ** (-1.0 / 3.0)


class FirsovScreeningLength(ThomasFermiScreeningLength):
    """Радиус экранирования Фирсова: a = 0,8853 a_B (Z1^1/2 + Z2^1/2)^(-2/3)."""

    name = "Фирсова"

    def __init__(self):
        super().__init__(x=1.0 / 2.0, y=2.0)


class LindhardScharffScreeningLength(ThomasFermiScreeningLength):
    """Радиус экранирования Линдхарда–Шарфа: a = 0,8853 a_B (Z1^2/3 + Z2^2/3)^(-1/2)."""

    name = "Линдхарда-Шарфа"

    def __init__(self):
        super().__init__(x=2.0 / 3.0, y=3.0 / 2.0)


class ZBLScreeningLength(ScreeningLength):
    """Универсальная длина экранирования: a_U = 0,8853 a_B (Z1^0,23 + Z2^0,23)^(-1)."""

    name = "ZBL"

    def __call__(self, z1: int, z2: int) -> float:
        return THOMAS_FERMI_CONSTANT * A_BOHR / (z1**0.23 + z2**0.23)
