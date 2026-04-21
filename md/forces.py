"""
Interacciones de par Lennard-Jones en 2D.

  V(r) = 4ε [ (σ/r)^12 - (σ/r)^6 ]

Unidades reducidas: σ = ε = m = kB = 1.

Derivación del signo:
  F_i = -∇_{r_i} V = (dV/dr) · (r_j - r_i)/r

  dV/dr = 4·(-12·r^{-13} + 6·r^{-7})

  F_i = (dV/dr / r) · (r_j - r_i)
       = 4·(-12·r^{-14} + 6·r^{-8}) · dr
       = (-48·ir12 + 24·ir6) · ir2 · dr

  con  dr = r_j - r_i,  ir2 = 1/r².
"""

import numpy as np


def compute_forces(pos: np.ndarray, r_cut: float = 2.5):
    """
    Calcula fuerzas LJ y energía potencial para N partículas en 2D.
    Sin condiciones de contorno periódicas (caja con paredes).

    Parameters
    ----------
    pos   : (N, 2) array — posiciones.
    r_cut : float        — radio de corte en unidades reducidas.

    Returns
    -------
    forces : (N, 2) ndarray
    U      : float — energía potencial total (shifted en el corte).
    """
    N      = pos.shape[0]
    forces = np.zeros_like(pos)
    U      = 0.0

    rc2   = r_cut * r_cut
    V_cut = 4.0 * (r_cut**(-12) - r_cut**(-6))

    for i in range(N - 1):
        dr = pos[i + 1:] - pos[i]           # (N-i-1, 2)  dr = r_j - r_i
        r2 = dr[:, 0]**2 + dr[:, 1]**2

        within = r2 < rc2
        if not within.any():
            continue

        r2w  = r2[within]
        drw  = dr[within]

        ir2  = 1.0 / r2w
        ir6  = ir2 * ir2 * ir2
        ir12 = ir6 * ir6

        # F_i = (-48·ir12 + 24·ir6)·ir2 · dr   (ver derivación en cabecera)
        fmag = (-48.0 * ir12 + 24.0 * ir6) * ir2

        fvec = fmag[:, np.newaxis] * drw          # F_i para cada par
        forces[i] += fvec.sum(axis=0)
        np.add.at(forces, np.where(within)[0] + (i + 1), -fvec)  # F_j = -F_i

        U += (4.0 * (ir12 - ir6) - V_cut).sum()

    return forces, U
