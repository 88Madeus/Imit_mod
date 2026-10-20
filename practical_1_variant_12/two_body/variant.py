"""Исходные данные варианта 12: А = 3 (Kr-C, Фирсов), Б = 5 (E0 = 1 эВ, Ar -> Ag)."""

from dataclasses import dataclass

from .kinematics import BinaryCollision
from .particles import ARGON, SILVER, Particle
from .potentials import ScreenedCoulombPotential
from .screening import KR_C, FirsovScreeningLength, ScreeningFunction, ScreeningLength


@dataclass(frozen=True)
class Variant:
    """Вариант задания: потенциал, радиус экранирования, энергия и пара частиц."""

    number: int
    screening_function: ScreeningFunction
    screening_length: ScreeningLength
    energy: float  # E0, эВ
    projectile: Particle
    target: Particle

    def build_collision(self) -> BinaryCollision:
        return BinaryCollision(self.projectile, self.target, self.energy)

    def build_potential(self) -> ScreenedCoulombPotential:
        return ScreenedCoulombPotential(
            self.projectile.z, self.target.z, self.screening_function, self.screening_length
        )

    def describe(self) -> str:
        collision = self.build_collision()
        potential = self.build_potential()
        return "\n".join(
            [
                f"Вариант {self.number}",
                f"  Потенциал: {potential.name}",
                f"  Налетающая частица: {self.projectile}",
                f"  Покоящаяся частица: {self.target}",
                f"  E0 = {self.energy:g} эВ, Er = {collision.relative_energy:.6g} эВ",
                f"  Длина экранирования a = {potential.a:.6g} нм",
            ]
        )


VARIANT_12 = Variant(
    number=12,
    screening_function=KR_C,
    screening_length=FirsovScreeningLength(),
    energy=1.0,
    projectile=ARGON,
    target=SILVER,
)
