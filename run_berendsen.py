"""
Parte 1: Simulación con termostato de Berendsen.

Se equilibra el sistema a temperatura T_TARGET, y en producción
se mide:
  (a) la distribución de momentos de una partícula aleatoria y se
      comprueba que sigue Maxwell-Boltzmann,
  (b) la energía de interacción media por partícula y su dispersión.
"""

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from md import Simulation, WallConfig, maxwell_boltzmann_check, mean_interaction_energy
from md.analysis import instant_temperature


def main():
    rng_select = np.random.default_rng(cfg.SEED)
    part_idx   = int(rng_select.integers(0, cfg.N))  # partícula aleatoria

    print("=" * 60)
    print("  Simulación MD — Termostato de Berendsen")
    print(f"  N={cfg.N}  L={cfg.L}  rho={cfg.RHO:.4f}  T={cfg.T_TARGET}")
    print(f"  Partícula a analizar: {part_idx}")
    print("=" * 60)

    sim = Simulation(
        N    = cfg.N,
        L    = cfg.L,
        T_init = cfg.T_TARGET,
        dt   = cfg.DT,
        r_cut = cfg.R_CUT,
        seed = cfg.SEED,
    )
    wall_cfg = WallConfig(cfg.L)   # paredes elásticas (Berendsen controla T)

    # ---- Equilibración ----
    print("\n[Equilibración]")
    sim.run(
        n_steps   = cfg.N_EQUIL,
        wall_cfg  = wall_cfg,
        thermostat = 'berendsen',
        T_target  = cfg.T_TARGET,
        tau_beren = cfg.TAU_BEREN,
        n_save    = cfg.N_SAVE,
        verbose   = True,
        verbose_interval = 5000,
    )
    sim.reset_histories()
    print(f"  Equilibración completada en {sim.step} pasos.")

    # ---- Producción ----
    print("\n[Producción]")
    sim.run(
        n_steps   = cfg.N_PROD,
        wall_cfg  = wall_cfg,
        thermostat = 'berendsen',
        T_target  = cfg.T_TARGET,
        tau_beren = cfg.TAU_BEREN,
        n_save    = cfg.N_SAVE,
        verbose   = True,
        verbose_interval = 10000,
    )
    sim.summary()

    # ---- Análisis de momentos ----
    # Serie temporal de velocidades de la partícula part_idx
    vel_part = np.array([snap[part_idx] for snap in sim.vel_history])  # (M, 2)

    fig_mb, ks_px, ks_v = maxwell_boltzmann_check(
        vel_part, T=cfg.T_TARGET, label='Berendsen'
    )
    fig_mb.suptitle(
        f'Distribución de velocidades — partícula {part_idx}  '
        f'(Berendsen, T={cfg.T_TARGET})',
        y=1.02
    )
    fig_mb.savefig('resultados_berendsen_MB.png', dpi=150, bbox_inches='tight')
    print(f"\n  KS-test  p_x : {ks_px:.4f}   |v| : {ks_v:.4f}")
    print("  (valores < 0.05 rechazarían la hipótesis de MB al 5%)")

    # ---- Energía de interacción ----
    mean_U, std_U = mean_interaction_energy(sim.potential_history)
    print(f"\n  Energía de interacción por partícula:")
    print(f"    <U>/N = {mean_U/cfg.N:.4f}  ±  {std_U/cfg.N:.4f}")

    # ---- Figuras de diagnóstico ----
    fig_hist, ax_hist = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    sim.plot_histories(ax=ax_hist)
    fig_hist.savefig('resultados_berendsen_series.png', dpi=150, bbox_inches='tight')

    fig_snap, ax_snap = plt.subplots(figsize=(5, 5))
    sim.snapshot_plot(ax=ax_snap, title=f'Snapshot final — Berendsen T={cfg.T_TARGET}')
    fig_snap.savefig('resultados_berendsen_snapshot.png', dpi=150, bbox_inches='tight')

    plt.show()
    print("\nFiguras guardadas en el directorio actual.")


if __name__ == '__main__':
    main()
