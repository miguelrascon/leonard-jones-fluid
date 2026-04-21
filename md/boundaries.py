"""
Condiciones de contorno en las paredes del recinto.

Dos tipos:
  - Elástica (adiabática): reflexión especular, |v_perp| no cambia.
  - Térmica: la partícula sale con velocidad muestreada de la
    distribución de Maxwell-Boltzmann a la temperatura T_pared.

Para la pared térmica en 2D (unidades reducidas, m=kB=1):
  - Componente tangencial: v_t ~ N(0, T)   [distribución normal]
  - Componente normal (hacia el interior): muestreada de la distribución
    de flujo  p(v_n) ∝ v_n * exp(-v_n^2 / 2T),  v_n > 0
    que es una distribución de Rayleigh con parámetro σ = sqrt(T).
    Muestra:  v_n = sqrt(-2*T * log(U)),  U ~ Uniforme(0,1).

Esta distribución es la correcta para una pared en equilibrio con un
baño térmico: garantiza la distribución Maxwell-Boltzmann en el bulk
a temperatura T (condición de balance detallado).
"""

import numpy as np


# Índices de pared (usados internamente)
_LEFT   = 0   # x = 0
_RIGHT  = 1   # x = L
_BOTTOM = 2   # y = 0
_TOP    = 3   # y = L


class WallConfig:
    """
    Configuración de las cuatro paredes del recinto.

    Parámetros
    ----------
    L      : float — lado de la caja.
    left   : str   — 'elastic' o 'thermal'.
    right  : str
    bottom : str
    top    : str
    T_left, T_right, T_bottom, T_top : float — temperatura de la pared
        (sólo relevante si la pared es térmica).
    """

    def __init__(self, L, left='elastic', right='elastic',
                 bottom='elastic', top='elastic',
                 T_left=1.0, T_right=1.0,
                 T_bottom=1.0, T_top=1.0):
        self.L    = L
        self.kind = {
            'left':   left,
            'right':  right,
            'bottom': bottom,
            'top':    top,
        }
        self.T = {
            'left':   T_left,
            'right':  T_right,
            'bottom': T_bottom,
            'top':    T_top,
        }

    def all_elastic(self):
        return all(k == 'elastic' for k in self.kind.values())


def _sample_thermal_velocities_1d_normal(n, T, rng):
    """Componente tangencial: distribución normal centrada en 0."""
    return rng.normal(0.0, np.sqrt(T), size=n)


def _sample_thermal_velocities_1d_flux(n, T, rng):
    """Componente normal (saliente): distribución de flujo Rayleigh."""
    U = rng.uniform(0.0, 1.0, size=n)
    # p(v) ∝ v*exp(-v²/2T)  →  v = sqrt(-2T·ln U)
    return np.sqrt(-2.0 * T * np.log(np.clip(U, 1e-14, None)))


def apply_walls(pos, vel, wall_cfg, rng):
    """
    Aplica condiciones de contorno después de cada paso de integración.

    Modifica pos y vel in-place y devuelve el número de colisiones con
    cada pared (útil para diagnósticos).

    Parameters
    ----------
    pos      : (N, 2) ndarray — posiciones.
    vel      : (N, 2) ndarray — velocidades.
    wall_cfg : WallConfig.
    rng      : np.random.Generator.

    Returns
    -------
    n_collisions : dict con keys 'left','right','bottom','top'.
    """
    L = wall_cfg.L
    n_col = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}

    # --- Pared izquierda (x = 0) ---
    hit = pos[:, 0] < 0.0
    if hit.any():
        n_col['left'] += hit.sum()
        pos[hit, 0] = -pos[hit, 0]          # reflexión de posición
        if wall_cfg.kind['left'] == 'elastic':
            vel[hit, 0] = np.abs(vel[hit, 0])
        else:
            T = wall_cfg.T['left']
            nh = hit.sum()
            vel[hit, 0] = _sample_thermal_velocities_1d_flux(nh, T, rng)
            vel[hit, 1] = _sample_thermal_velocities_1d_normal(nh, T, rng)

    # --- Pared derecha (x = L) ---
    hit = pos[:, 0] > L
    if hit.any():
        n_col['right'] += hit.sum()
        pos[hit, 0] = 2.0 * L - pos[hit, 0]
        if wall_cfg.kind['right'] == 'elastic':
            vel[hit, 0] = -np.abs(vel[hit, 0])
        else:
            T = wall_cfg.T['right']
            nh = hit.sum()
            vel[hit, 0] = -_sample_thermal_velocities_1d_flux(nh, T, rng)
            vel[hit, 1] = _sample_thermal_velocities_1d_normal(nh, T, rng)

    # --- Pared inferior (y = 0) ---
    hit = pos[:, 1] < 0.0
    if hit.any():
        n_col['bottom'] += hit.sum()
        pos[hit, 1] = -pos[hit, 1]
        if wall_cfg.kind['bottom'] == 'elastic':
            vel[hit, 1] = np.abs(vel[hit, 1])
        else:
            T = wall_cfg.T['bottom']
            nh = hit.sum()
            vel[hit, 1] = _sample_thermal_velocities_1d_flux(nh, T, rng)
            vel[hit, 0] = _sample_thermal_velocities_1d_normal(nh, T, rng)

    # --- Pared superior (y = L) ---
    hit = pos[:, 1] > L
    if hit.any():
        n_col['top'] += hit.sum()
        pos[hit, 1] = 2.0 * L - pos[hit, 1]
        if wall_cfg.kind['top'] == 'elastic':
            vel[hit, 1] = -np.abs(vel[hit, 1])
        else:
            T = wall_cfg.T['top']
            nh = hit.sum()
            vel[hit, 1] = -_sample_thermal_velocities_1d_flux(nh, T, rng)
            vel[hit, 0] = _sample_thermal_velocities_1d_normal(nh, T, rng)

    return n_col
