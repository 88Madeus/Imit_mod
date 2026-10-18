"""Варианты расчётного задания практической работы № 1."""

from dataclasses import dataclass

from .kinematics import BinaryCollision
from .particles import ParticleCatalog
from .potentials import ScreenedCoulombPotential
from .screening import (
    BOHR,
    KR_C,
    MOLIERE,
    ZBL,
    FirsovScreeningLength,
    LindhardScharffScreeningLength,
    ScreeningFunction,
    ScreeningLength,
    ZBLScreeningLength,
)


@dataclass(frozen=True)
class PotentialOption:
    """Столбец таблицы «А»: потенциал и радиус экранирования."""

    screening_function: ScreeningFunction
    screening_length_type: type


@dataclass(frozen=True)
class CollisionOption:
    """Столбец таблицы «Б»: энергия и пара сталкивающихся частиц."""

    energy: float  # E0, эВ
    projectile: str
    target: str


POTENTIAL_OPTIONS = {
    1: PotentialOption(BOHR, FirsovScreeningLength),
    2: PotentialOption(MOLIERE, LindhardScharffScreeningLength),
    3: PotentialOption(KR_C, FirsovScreeningLength),
    4: PotentialOption(ZBL, ZBLScreeningLength),
}

COLLISION_OPTIONS = {
    1: CollisionOption(0.1, "p", "Cu"),
    2: CollisionOption(0.5, "He2", "Au"),
    3: CollisionOption(0.3, "He", "Al"),
    4: CollisionOption(0.8, "p", "Si"),
    5: CollisionOption(1.0, "Ar", "Ag"),
    6: CollisionOption(1.5, "p", "Nb"),
}

# № варианта -> (А, Б)
VARIANT_TABLE = {
    1: (1, 6), 2: (4, 5), 3: (4, 2), 4: (3, 1), 5: (4, 6), 6: (1, 2),
    7: (2, 5), 8: (2, 3), 9: (4, 4), 10: (1, 1), 11: (3, 6), 12: (3, 5),
    13: (4, 1), 14: (1, 5), 15: (2, 4), 16: (2, 3), 17: (3, 4), 18: (2, 6),
    19: (4, 3), 20: (2, 1), 21: (2, 2), 22: (3, 2), 23: (3, 3), 24: (1, 4),
}


@dataclass(frozen=True)
class Variant:
    """Индивидуальный вариант задания."""

    number: int
    potential_index: int
    collision_index: int

    @classmethod
    def from_number(cls, number: int) -> "Variant":
        if number not in VARIANT_TABLE:
            raise ValueError(f"Вариант должен быть от 1 до {len(VARIANT_TABLE)}")
        return cls(number, *VARIANT_TABLE[number])

    @property
    def potential_option(self) -> PotentialOption:
        return POTENTIAL_OPTIONS[self.potential_index]

    @property
    def collision_option(self) -> CollisionOption:
        return COLLISION_OPTIONS[self.collision_index]

    def build_collision(self) -> BinaryCollision:
        option = self.collision_option
        return BinaryCollision(
            ParticleCatalog.get(option.projectile),
            ParticleCatalog.get(option.target),
            option.energy,
        )

    def build_potential(self) -> ScreenedCoulombPotential:
        collision = self.build_collision()
        option = self.potential_option
        screening_length: ScreeningLength = option.screening_length_type()
        return ScreenedCoulombPotential(
            collision.projectile.z,
            collision.target.z,
            option.screening_function,
            screening_length,
        )

    def describe(self) -> str:
        collision = self.build_collision()
        potential = self.build_potential()
        return "\n".join(
            [
                f"Вариант {self.number}: А = {self.potential_index}, Б = {self.collision_index}",
                f"  Потенциал: {potential.name}",
                f"  Налетающая частица: {collision.projectile}",
                f"  Покоящаяся частица: {collision.target}",
                f"  E0 = {collision.energy:g} эВ, Er = {collision.relative_energy:.6g} эВ",
                f"  Длина экранирования a = {potential.a:.6g} нм",
            ]
        )
