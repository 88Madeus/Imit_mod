"""Подготовка рисунков и таблиц для LaTeX-отчёта по практическому заданию № 1.

Запуск из каталога practical_1:  python report/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from two_body import TwoBodyScatteringExperiment, Variant  # noqa: E402
from two_body.plotting import ResultsPlotter  # noqa: E402
from two_body.potentials import E2, CoulombPotential  # noqa: E402
from two_body.scattering import (  # noqa: E402
    ClosestApproachSolver,
    GaussMehlerScatteringAngle,
    RadialFunction,
)

REPORT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = REPORT_DIR / "figures"
GENERATED_DIR = REPORT_DIR / "generated"
NODE_COUNTS = (10, 25, 50, 100, 200, 400)


def num(value: float, digits: int = 4) -> str:
    """Число с десятичной запятой для вставки в LaTeX."""
    return f"{value:.{digits}f}".replace(".", "{,}")


def sci(value: float) -> str:
    """Число в экспоненциальной записи: 3,21 \\cdot 10^{-5}."""
    mantissa, exponent = f"{value:.2e}".split("e")
    return f"${mantissa.replace('.', '{,}')} \\cdot 10^{{{int(exponent)}}}$"


class CoulombBenchmark:
    """Тестовый расчёт: кулоновский потенциал, для которого известно точное решение."""

    def __init__(self, z1: int = 1, z2: int = 29, relative_energy: float = 100.0):
        self.potential = CoulombPotential(z1, z2)
        self.b = z1 * z2 * E2 / relative_energy
        self.g = RadialFunction(self.potential, relative_energy)
        self.solver = ClosestApproachSolver(self.g)
        self.rho = np.linspace(0.01, 2.0, 200) * self.b
        self.closest_approach = self.solver.solve_many(self.rho)

    def exact_chi(self):
        return 2 * np.arctan(self.b / (2 * self.rho))

    def numerical_chi(self, n_nodes: int):
        angle = GaussMehlerScatteringAngle(self.g, n_nodes)
        return angle.compute_many(self.rho, self.closest_approach)

    def max_error(self, n_nodes: int) -> float:
        return float(np.max(np.abs(self.numerical_chi(n_nodes) - self.exact_chi())))


class ReportArtifacts:
    """Формирует рисунки и фрагменты таблиц отчёта."""

    def __init__(self, variant_number: int = 4):
        self.variant = Variant.from_number(variant_number)
        self.results = TwoBodyScatteringExperiment(self.variant).run()
        self.coulomb = CoulombBenchmark()

    def variant_figures(self) -> None:
        ResultsPlotter(self.results, FIGURES_DIR, dpi=200, show_titles=False).plot_all()

    def coulomb_figure(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
        rho = self.coulomb.rho / self.coulomb.b
        ax1.plot(rho, np.degrees(self.coulomb.exact_chi()), label="точное решение")
        ax1.plot(rho[::8], np.degrees(self.coulomb.numerical_chi(100))[::8], "o",
                 ms=4, label="Гаусс–Мелер, N = 100")
        ax1.set_xlabel("ρ / b")
        ax1.set_ylabel("χ, град")
        ax1.legend()
        errors = [self.coulomb.max_error(n) for n in NODE_COUNTS]
        ax2.loglog(NODE_COUNTS, errors, "o-", label="максимальная погрешность")
        ax2.loglog(NODE_COUNTS, errors[0] * (NODE_COUNTS[0] / np.array(NODE_COUNTS)) ** 2,
                   "--", color="gray", label="∝ 1/N²")
        ax2.set_xlabel("N")
        ax2.set_ylabel("max |Δχ|, рад")
        ax2.legend()
        for ax in (ax1, ax2):
            ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "test_coulomb.png", dpi=200)
        plt.close(fig)

    def convergence_table(self) -> None:
        reference = TwoBodyScatteringExperiment(self.variant, n_nodes=1600).run().chi
        rows = []
        for n in NODE_COUNTS:
            chi = TwoBodyScatteringExperiment(self.variant, n_nodes=n).run().chi
            rows.append(f"{n} & {sci(self.coulomb.max_error(n))} & "
                        f"{sci(float(np.max(np.abs(chi - reference))))} \\\\ \\hline")
        (GENERATED_DIR / "convergence_rows.tex").write_text("\n".join(rows) + "\n",
                                                            encoding="utf-8")

    def results_table(self) -> None:
        res = self.results
        rows = []
        for i in [0] + list(range(19, len(res.rho), 20)):
            rows.append(
                f"{num(res.rho[i], 5)} & {num(res.closest_approach[i], 5)} & "
                f"{num(np.degrees(res.chi[i]), 2)} & {num(np.degrees(res.theta1[i]), 2)} & "
                f"{num(np.degrees(res.theta2[i]), 2)} \\\\ \\hline"
            )
        (GENERATED_DIR / "results_rows.tex").write_text("\n".join(rows) + "\n",
                                                        encoding="utf-8")

    def build(self) -> None:
        # Крупный шрифт: рисунки уменьшаются до ширины полосы набора
        plt.rcParams.update({"font.size": 16})
        FIGURES_DIR.mkdir(exist_ok=True)
        GENERATED_DIR.mkdir(exist_ok=True)
        self.variant_figures()
        self.coulomb_figure()
        self.convergence_table()
        self.results_table()


if __name__ == "__main__":
    ReportArtifacts().build()
    print(f"Рисунки: {FIGURES_DIR}\nТаблицы: {GENERATED_DIR}")
