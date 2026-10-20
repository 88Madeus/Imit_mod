"""Частицы, участвующие в столкновении."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Particle:
    """Частица (ион или атом): обозначение, заряд ядра Z и масса в а.е.м."""

    symbol: str
    name: str
    z: int
    mass: float

    def __str__(self) -> str:
        return f"{self.symbol} (Z = {self.z}, M = {self.mass:g} а.е.м.)"


ARGON = Particle("Ar", "аргон", 18, 39.948)
SILVER = Particle("Ag", "серебро", 47, 107.8682)
