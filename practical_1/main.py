"""Запуск расчёта практического задания № 1 (по умолчанию — вариант 4)."""

import argparse
from pathlib import Path

import numpy as np

from two_body import TwoBodyScatteringExperiment, Variant
from two_body.plotting import ResultsPlotter


class Application:
    """Консольное приложение: расчёт, вывод сводки, сохранение графиков и таблицы."""

    def __init__(self, args: argparse.Namespace):
        self.variant = Variant.from_number(args.variant)
        self.output_dir = Path(args.output) / f"variant_{self.variant.number:02d}"
        self.experiment = TwoBodyScatteringExperiment(
            self.variant, n_nodes=args.nodes, n_rho=args.points
        )

    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--variant", type=int, default=4, help="номер варианта (1–24)")
        parser.add_argument("--nodes", type=int, default=100,
                            help="число узлов квадратуры Гаусса–Мелера")
        parser.add_argument("--points", type=int, default=200,
                            help="число точек сетки прицельных параметров")
        parser.add_argument("--output", default=str(Path(__file__).parent / "results"),
                            help="каталог для результатов")
        return parser.parse_args()

    def run(self) -> None:
        print(self.variant.describe())
        results = self.experiment.run()

        print(f"\nρmax = {results.rho_max:.6g} нм, "
              f"Δρ = {results.rho_max / len(results.rho):.6g} нм")
        print(f"R(ρ → 0) = {self.experiment.solver(0.0):.6g} нм")
        print(f"\n{'ρ, нм':>12} {'R, нм':>12} {'χ, °':>10} {'θ1, °':>10} {'θ2, °':>10}")
        for i in np.linspace(0, len(results.rho) - 1, 11, dtype=int):
            print(f"{results.rho[i]:12.6f} {results.closest_approach[i]:12.6f} "
                  f"{np.degrees(results.chi[i]):10.4f} {np.degrees(results.theta1[i]):10.4f} "
                  f"{np.degrees(results.theta2[i]):10.4f}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        results.save_csv(self.output_dir / "scattering_table.csv")
        paths = ResultsPlotter(results, self.output_dir).plot_all()
        print(f"\nРезультаты сохранены в {self.output_dir}:")
        for path in paths:
            print(f"  {path.name}")


if __name__ == "__main__":
    Application(Application.parse_args()).run()
