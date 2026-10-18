"""Частицы, участвующие в парных столкновениях."""

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


class ParticleCatalog:
    """Справочник частиц, встречающихся в вариантах задания."""

    _particles = {
        p.symbol: p
        for p in (
            Particle("p", "протон", 1, 1.007276),
            Particle("He", "гелий", 2, 4.002602),
            Particle("He2", "гелий (He2+)", 2, 4.002602),
            Particle("Ar", "аргон", 18, 39.948),
            Particle("Al", "алюминий", 13, 26.981539),
            Particle("Si", "кремний", 14, 28.0855),
            Particle("Cu", "медь", 29, 63.546),
            Particle("Nb", "ниобий", 41, 92.90637),
            Particle("Ag", "серебро", 47, 107.8682),
            Particle("Au", "золото", 79, 196.966569),
        )
    }

    @classmethod
    def get(cls, symbol: str) -> Particle:
        try:
            return cls._particles[symbol]
        except KeyError:
            raise KeyError(f"Неизвестная частица: {symbol!r}") from None
