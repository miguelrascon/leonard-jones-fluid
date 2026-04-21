"""
Parte 3: Gradiente de temperatura — transporte de calor.

Configuración:
  - Pared inferior (y = 0)  →  temperatura T_HOT
  - Pared superior (y = L)  →  temperatura T_COLD = T_HOT / 2
  - Paredes laterales (x = 0, x = L)  →  elásticas (adiabáticas)

El sistema no termaliza a una T uniforme: se establece un gradiente
estacionario de temperatura a lo largo de y.  Se estima T(x,y)
localmente a partir de la densidad de energía cinética en celdas de
área δΣ = (L/nx) × (L/ny).

Uso:
  python run_gradient.py

Los resultados se guardan como 'resultados_gradiente_*.png'.
"""

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from md import Simulation, WallConfig
from md.analysis import temperature_field, plot_temperature_field


# -----------------------------------------------------------------------
# Parámetros específicos de esta parte
# -----------------------------------------------------------------------
T_HOT   = cfg.T_TARGET        # pared inferior
T_COLD  = cfg.T_TARGET / 2.0  # pared superior

N_EQUIL_GRAD = 50_000   # equilibración más larga (el gradiente tarda en establecerse)
N_PROD_GRAD  = 100_000
N_SAVE_GRAD  = 100

NX = cfg.N_CELLS_X
NY = cfg.N_CELLS_Y


def main():
    print("=" * 60)
    print("  Simulación MD — Gradiente de temperatura")
    print(f"  N={cfg.N}  L={cfg.L}  T_hot={T_HOT}  T_cold={T_COLD}")
    print("=" * 60)

    sim = Simulation(
        N      = cfg.N,
        L      = cfg.L,
        T_init = 0.5 * (T_HOT + T_COLD),   # inicializar a T media
        dt     = cfg.DT,
        r_cut  = cfg.R_CUT,
        seed   = cfg.SEED + 99,
    )

    wall_cfg = WallConfig(
        cfg.L,
        left   = 'elastic',
        right  = 'elastic',
        bottom = 'thermal',   # y = 0  →  caliente
        top    = 'thermal',   # y = L  →  fría
        T_bottom = T_HOT,
        T_top    = T_COLD,
    )

    # ---- Equilibración ----
    print("\n[Equilibración — establecimiento del régimen estacionario]")
    sim.run(
        n_steps  = N_EQUIL_GRAD,
        wall_cfg = wall_cfg,
        thermostat = 'none',
        n_save   = N_SAVE_GRAD,
        verbose  = True,
        verbose_interval = 10000,
    )
    sim.reset_histories()
    print("  Régimen estacionario alcanzado.")

    # ---- Producción ----
    print("\n[Producción — muestreo del campo de temperatura]")
    sim.run(
        n_steps  = N_PROD_GRAD,
        wall_cfg = wall_cfg,
        thermostat = 'none',
        n_save   = N_SAVE_GRAD,
        verbose  = True,
        verbose_interval = 20000,
    )
    sim.summary()

    # ---- Campo de temperatura ----
    print("\n[Calculando T(x,y)...]")
    pos_arr = np.array(sim.pos_history)   # (M, N, 2)
    vel_arr = np.array(sim.vel_history)

    T_map, x_edges, y_edges = temperature_field(
        pos_arr, vel_arr, cfg.L, NX, NY
    )

    # ---- Perfil medio T(y) ----
    # Promediamos T_map sobre la dirección x para obtener T(y)
    T_profile_y = np.nanmean(T_map, axis=0)   # (NY,)
    y_centers   = 0.5 * (y_edges[:-1] + y_edges[1:])

    # ---- Figuras ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Mapa 2D
    plot_temperature_field(
        T_map, x_edges, y_edges, ax=axes[0],
        title=f'T(x,y)  —  gradiente  [T_hot={T_HOT}, T_cold={T_COLD}]'
    )

    # Perfil 1D T(y)
    axes[1].plot(y_centers, T_profile_y, 'o-', ms=5, lw=1.5, label='MD')
    axes[1].axhline(T_HOT,  color='red',  ls='--', lw=1, label=f'T_hot = {T_HOT}')
    axes[1].axhline(T_COLD, color='blue', ls='--', lw=1, label=f'T_cold = {T_COLD}')
    # Perfil lineal teórico (Fourier)
    y_th = np.linspace(0, cfg.L, 200)
    T_th = T_HOT + (T_COLD - T_HOT) / cfg.L * y_th
    axes[1].plot(y_th, T_th, 'k:', lw=2, label='Fourier (lineal)')
    axes[1].set_xlabel('y')
    axes[1].set_ylabel('T  (unid. reducidas)')
    axes[1].set_title('Perfil de temperatura T(y)')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig('resultados_gradiente.png', dpi=150, bbox_inches='tight')
    print("  Figura guardada en 'resultados_gradiente.png'.")

    # Guardar datos numéricos para análisis posterior
    np.savetxt('T_map.dat', T_map,
               header=f'Campo T(x,y)  NX={NX} NY={NY}  L={cfg.L}')
    np.savetxt('T_profile_y.dat', np.column_stack([y_centers, T_profile_y]),
               header='y  T(y)')
    print("  Datos guardados en 'T_map.dat' y 'T_profile_y.dat'.")

    plt.show()


if __name__ == '__main__':
    main()
