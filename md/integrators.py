"""
Integrador de Velocity Verlet para la dinámica molecular en 2D.

El algoritmo es:
    r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    a(t+dt) = F(r(t+dt)) / m
    v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt

Es simpléctico, conserva exactamente una energía modificada (sombra),
y es reversible en el tiempo — propiedades importantes para MD correcta.
"""

import numpy as np
from .forces import compute_forces


def velocity_verlet_step(pos, vel, forces, dt, r_cut=2.5, mass=1.0):
    """
    Avanza el sistema un paso temporal dt.

    Parameters
    ----------
    pos    : (N, 2) — posiciones en t.
    vel    : (N, 2) — velocidades en t.
    forces : (N, 2) — fuerzas en t (ya calculadas).
    dt     : float  — paso temporal.
    r_cut  : float  — radio de corte LJ.
    mass   : float  — masa de las partículas.

    Returns
    -------
    pos_new    : (N, 2)
    vel_new    : (N, 2)
    forces_new : (N, 2)
    U_new      : float — energía potencial en t+dt.
    """
    acc = forces / mass

    # Paso de posición
    pos_new = pos + vel * dt + 0.5 * acc * dt * dt

    # Fuerzas en la nueva posición (antes de aplicar paredes para que
    # los vecinos sean coherentes — las paredes se aplican después fuera)
    forces_new, U_new = compute_forces(pos_new, r_cut)

    acc_new = forces_new / mass

    # Paso de velocidad
    vel_new = vel + 0.5 * (acc + acc_new) * dt

    return pos_new, vel_new, forces_new, U_new
