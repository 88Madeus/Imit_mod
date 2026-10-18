"""Построение графиков по результатам расчёта."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .experiment import ScatteringResults  # noqa: E402


class ResultsPlotter:
    """Строит и сохраняет графики пп. 1–4 задания."""

    def __init__(self, results: ScatteringResults, output_dir: Path, dpi: int = 150):
        self.results = results
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        collision = results.collision
        self.subtitle = (
            f"{collision.projectile.symbol} → {collision.target.symbol}, "
            f"E0 = {collision.energy:g} эВ, {results.potential.name}"
        )

    def _new_figure(self, title: str):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.set_title(f"{title}\n{self.subtitle}", fontsize=11)
        ax.grid(True, which="both", alpha=0.3)
        return fig, ax

    def _save(self, fig, name: str) -> Path:
        path = self.output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi)
        plt.close(fig)
        return path

    def plot_potential(self) -> Path:
        res = self.results
        fig, ax = self._new_figure("Потенциальная энергия взаимодействия U(r)")
        ax.loglog(res.r, res.potential_energy, label="U(r)")
        ax.axhline(res.collision.relative_energy, color="tab:red", ls="--",
                   label=f"Er = {res.collision.relative_energy:.4g} эВ")
        ax.set_xlabel("r, нм")
        ax.set_ylabel("U, эВ")
        ax.legend()
        return self._save(fig, "1a_potential.png")

    def plot_kinematic_factor(self) -> Path:
        res = self.results
        fig, (ax_k, ax_e) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Кинематический фактор k(θ1) и энергия E1(θ1)\n{self.subtitle}",
                     fontsize=11)
        angles = np.degrees(res.theta1_grid)
        ax_k.plot(angles, res.kinematic_factor, color="tab:blue")
        ax_k.set_ylabel("k = E1 / E0")
        ax_e.plot(angles, res.scattered_energy, color="tab:orange")
        ax_e.set_ylabel("E1, эВ")
        for ax in (ax_k, ax_e):
            ax.set_xlabel("θ1, град")
            ax.grid(True, alpha=0.3)
        return self._save(fig, "1b_kinematic_factor.png")

    def plot_closest_approach(self) -> Path:
        res = self.results
        fig, ax = self._new_figure("Расстояние наименьшего сближения R(ρ)")
        ax.plot(res.rho, res.closest_approach, label="R(ρ)")
        ax.plot(res.rho, res.rho, color="gray", ls=":", label="R = ρ")
        ax.axvline(res.rho_max, color="tab:red", ls="--",
                   label=f"ρmax = {res.rho_max:.4g} нм")
        ax.set_xlabel("ρ, нм")
        ax.set_ylabel("R, нм")
        ax.legend()
        return self._save(fig, "2_closest_approach.png")

    def plot_scattering_angles(self) -> Path:
        res = self.results
        fig, ax = self._new_figure("Углы рассеяния χ(ρ), θ1(ρ), θ2(ρ)")
        ax.plot(res.rho, np.degrees(res.chi), label="χ (ц-система)")
        ax.plot(res.rho, np.degrees(res.theta1), ls="--", label="θ1 (л-система)")
        ax.plot(res.rho, np.degrees(res.theta2), ls="-.", label="θ2 (л-система)")
        ax.set_xlabel("ρ, нм")
        ax.set_ylabel("угол, град")
        ax.legend()
        return self._save(fig, "3_scattering_angles.png")

    def plot_impact_parameter(self) -> Path:
        res = self.results
        fig, ax = self._new_figure("Прицельный параметр ρ(θ1) и ρ(θ2)")
        cs1, cs2 = res.cross_section1, res.cross_section2
        ax.plot(np.degrees(cs1.theta), cs1.rho, label="ρ(θ1)")
        ax.plot(np.degrees(cs2.theta), cs2.rho, ls="--", label="ρ(θ2)")
        ax.set_xlabel("θ, град")
        ax.set_ylabel("ρ, нм")
        ax.legend()
        return self._save(fig, "4a_impact_parameter.png")

    def plot_cross_sections(self) -> Path:
        res = self.results
        fig, ax = self._new_figure("Дифференциальное сечение упругого рассеяния ω(θ)")
        cs1, cs2 = res.cross_section1, res.cross_section2
        ax.semilogy(np.degrees(cs1.theta), cs1.omega, label="ω(θ1)")
        ax.semilogy(np.degrees(cs2.theta), cs2.omega, ls="--", label="ω(θ2)")
        ax.set_xlabel("θ, град")
        ax.set_ylabel("ω, нм²/ср")
        ax.legend()
        return self._save(fig, "4b_cross_sections.png")

    def plot_all(self) -> list:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return [
            self.plot_potential(),
            self.plot_kinematic_factor(),
            self.plot_closest_approach(),
            self.plot_scattering_angles(),
            self.plot_impact_parameter(),
            self.plot_cross_sections(),
        ]
