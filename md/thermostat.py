"""
Termostato de Berendsen.

Las velocidades se reescalan en cada paso según:
    v → λ·v,    con    λ = sqrt( 1 + (dt/τ)·(T_ref/T_inst - 1) )

Este termostato no genera el ensemble canónico exacto (no respeta
fluctuaciones termodinámicas correctas), pero converge rápidamente
a la temperatura objetivo y es fácil de implementar.  Para la
comparación con las paredes térmicas basta con equilibrar a T dada.

Referencia: Berendsen et al., J. Chem. Phys. 81, 3684 (1984).
"""

import numpy as np
from .analysis import instant_temperature


def berendsen_rescale(vel, T_target, dt, tau, mass=1.0):
    """
    Reescala las velocidades según el termostato de Berendsen.

    Parameters
    ----------
    vel      : (N, 2) ndarray — velocidades actuales.
    T_target : float — temperatura objetivo.
    dt       : float — paso temporal.
    tau      : float — tiempo de acoplamiento.
    mass     : float — masa de las partículas.

    Returns
    -------
    vel : (N, 2) — velocidades reescaladas (modificadas in-place).
    lam : float  — factor de escala λ (útil para diagnóstico).
    """
    T_inst = instant_temperature(vel, mass)

    if T_inst < 1e-10:
        return vel, 1.0

    ratio = T_target / T_inst
    lam   = np.sqrt(1.0 + (dt / tau) * (ratio - 1.0))

    # Protección frente a reescalados excesivos en los primeros pasos
    lam = np.clip(lam, 0.5, 2.0)

    vel *= lam
    return vel, lam
