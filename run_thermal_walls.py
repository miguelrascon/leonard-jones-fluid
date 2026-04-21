"""
Parte 2: Termalización mediante paredes térmicas (sin termostato de Berendsen).

Las cuatro paredes están a temperatura T_TARGET.  Cuando una partícula
choca contra una pared, en lugar de rebotar elásticamente sale con una
velocidad muestreada de la distribución de Maxwell-Boltzmann a T_pared.

Se comparan los resultados con los del termostato de Berendsen:
  - distribución de velocidades,
  - temperatura media de producción.
"""

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from md import Simulation, WallConfig, maxwell_boltzmann_check, mean_interaction_energy


def main():
    rng_select = np.random.default_rng(cfg.SEED + 1)
    part_idx   = int(rng_select.integers(0, cfg.N))

    print("=" * 60)
    print("  Simulación MD — Paredes Térmicas (sin Berendsen)")
    print(f"  N={cfg.N}  L={cfg.L}  T_pared={cfg.T_TARGET}")
    print(f"  Partícula a analizar: {part_idx}")
    print("=" * 60)

    sim = Simulation(
        N      = cfg.N,
        L      = cfg.L,
        T_init = cfg.T_TARGET,
        dt     = cfg.DT,
        r_cut  = cfg.R_CUT,
        seed   = cfg.SEED + 1,
    )

    # Todas las paredes a T_TARGET
    wall_cfg = WallConfig(
        cfg.L,
        left='thermal',  right='thermal',
        bottom='thermal', top='thermal',
        T_left=cfg.T_TARGET, T_right=cfg.T_TARGET,
        T_bottom=cfg.T_TARGET, T_top=cfg.T_TARGET,
    )

    # ---- Equilibración ----
    print("\n[Equilibración]")
    sim.run(
        n_steps  = cfg.N_EQUIL,
        wall_cfg = wall_cfg,
        thermostat = 'none',
        n_save   = cfg.N_SAVE,
        verbose  = True,
        verbose_interval = 5000,
    )
    sim.reset_histories()
    print(f"  Equilibración completada en {sim.step} pasos.")

    # ---- Producción ----
    print("\n[Producción]")
    sim.run(
        n_steps  = cfg.N_PROD,
        wall_cfg = wall_cfg,
        thermostat = 'none',
        n_save   = cfg.N_SAVE,
        verbose  = True,
        verbose_interval = 10000,
    )
    sim.summary()

    # ---- Distribución de velocidades ----
    vel_part = np.array([snap[part_idx] for snap in sim.vel_history])

    fig_mb, ks_px, ks_v = maxwell_boltzmann_check(
        vel_part, T=cfg.T_TARGET, label='Paredes térmicas'
    )
    fig_mb.suptitle(
        f'Distribución de velocidades — partícula {part_idx}  '
        f'(Paredes térmicas, T={cfg.T_TARGET})',
        y=1.02
    )
    fig_mb.savefig('resultados_paredes_MB.png', dpi=150, bbox_inches='tight')
    print(f"\n  KS-test  p_x : {ks_px:.4f}   |v| : {ks_v:.4f}")

    # ---- Comparación de temperaturas ----
    T_arr = np.array(sim.temperature_history)
    print(f"\n  T media (paredes térmicas) = {T_arr.mean():.4f} ± {T_arr.std():.4f}")
    print(f"  T objetivo                 = {cfg.T_TARGET:.4f}")

    # ---- Energía de interacción ----
    mean_U, std_U = mean_interaction_energy(sim.potential_history)
    print(f"\n  <U>/N = {mean_U/cfg.N:.4f}  ±  {std_U/cfg.N:.4f}")

    # ---- Evolución temporal ----
    fig_hist, ax_hist = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    sim.plot_histories(ax=ax_hist)
    fig_hist.savefig('resultados_paredes_series.png', dpi=150, bbox_inches='tight')

    plt.show()
    print("\nFiguras guardadas en el directorio actual.")


if __name__ == '__main__':
    main()
