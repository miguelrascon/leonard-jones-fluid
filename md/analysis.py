"""
Herramientas de análisis para la simulación MD.

  - Temperatura instantánea
  - Energía de interacción media
  - Verificación de la distribución de Maxwell-Boltzmann
  - Campo de temperatura 2D (para el gradiente)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def instant_temperature(vel, mass=1.0):
    """
    Temperatura instantánea a partir del teorema de equipartición en 2D.

        T = <m·v²> / (2·kB) = m·<v²> / 2    (kB = 1, m·v² promediado sobre N)

    En unidades reducidas (m = kB = 1):
        T_inst = sum(v^2) / (2N - 2)
    donde el denominador son los grados de libertad traslacionales
    una vez eliminado el movimiento del CM (para coherencia con el
    termostato, que actúa sobre las velocidades totales).
    """
    N   = vel.shape[0]
    dof = 2 * N - 2     # grados de libertad en 2D sin CM
    return mass * np.sum(vel**2) / dof


def kinetic_energy(vel, mass=1.0):
    return 0.5 * mass * np.sum(vel**2)


def remove_cm_velocity(vel):
    """Elimina la velocidad del centro de masa."""
    vel -= vel.mean(axis=0)
    return vel


# ---------------------------------------------------------------------------
# Maxwell-Boltzmann
# ---------------------------------------------------------------------------

def maxwell_boltzmann_check(vel, T, mass=1.0, particle_idx=None, ax=None,
                             n_bins=30, label=''):
    """
    Comprueba si la distribución de momentos sigue Maxwell-Boltzmann.

    En 2D con m=kB=1 cada componente de velocidad sigue N(0, T),
    mientras que la rapidez |v| sigue f(v) = (v/T) exp(-v²/2T)
    (distribución de Maxwell en 2D, también llamada Rayleigh).

    Parameters
    ----------
    vel          : (N, 2) array  (o lista de snapshots (M, 2)) — si se
                   pasa un único snapshot se verifica una partícula.
    T            : float — temperatura esperada.
    particle_idx : int o None — si vel tiene shape (M, 2) (serie temporal
                   de una partícula), se interpreta directamente.
    ax           : lista de dos axes de matplotlib o None.
    label        : str — etiqueta para la leyenda.

    Returns
    -------
    fig, ks_stat_px, ks_stat_v : estadístico KS para p_x y |v|.
    """
    if particle_idx is not None:
        # vel es un array (M, 2): serie temporal de una partícula
        data = np.asarray(vel)   # (M, 2)
    else:
        data = vel               # (N, 2): snapshot global

    px   = data[:, 0]
    py   = data[:, 1]
    spd  = np.sqrt(px**2 + py**2)

    sigma = np.sqrt(T / mass)

    # Test KS
    ks_px, pval_px = stats.kstest(px, 'norm', args=(0, sigma))
    ks_py, pval_py = stats.kstest(py, 'norm', args=(0, sigma))
    ks_v,  pval_v  = stats.kstest(spd, lambda x: 1 - np.exp(-x**2 / (2 * T / mass)))

    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig = ax[0].get_figure()

    # Histograma de p_x
    ax[0].hist(px, bins=n_bins, density=True, alpha=0.6, label=f'MD {label}')
    xr = np.linspace(px.min(), px.max(), 300)
    ax[0].plot(xr, stats.norm.pdf(xr, 0, sigma),
               'r--', lw=2, label=f'MB  T={T:.2f}')
    ax[0].set_xlabel(r'$p_x$  (unid. reducidas)')
    ax[0].set_ylabel('densidad de probabilidad')
    ax[0].set_title(f'Componente $p_x$  |  KS = {ks_px:.3f}  (p={pval_px:.3f})')
    ax[0].legend()

    # Histograma de |v|
    ax[1].hist(spd, bins=n_bins, density=True, alpha=0.6, label=f'MD {label}')
    vr = np.linspace(0, spd.max() * 1.1, 300)
    # distribución de Rayleigh (Maxwell 2D)
    rayleigh_pdf = (vr / (T / mass)) * np.exp(-vr**2 / (2 * T / mass))
    ax[1].plot(vr, rayleigh_pdf, 'r--', lw=2, label=f'MB  T={T:.2f}')
    ax[1].set_xlabel(r'$|\mathbf{v}|$  (unid. reducidas)')
    ax[1].set_ylabel('densidad de probabilidad')
    ax[1].set_title(f'Rapidez $|v|$  |  KS = {ks_v:.3f}  (p={pval_v:.3f})')
    ax[1].legend()

    fig.tight_layout()

    return fig, ks_px, ks_v


# ---------------------------------------------------------------------------
# Energía de interacción
# ---------------------------------------------------------------------------

def mean_interaction_energy(U_series):
    """
    Calcula la energía de interacción media y su desviación estándar.

    Parameters
    ----------
    U_series : array-like — serie temporal de la energía potencial total.

    Returns
    -------
    mean_U : float
    std_U  : float
    """
    U  = np.asarray(U_series)
    return U.mean(), U.std()


# ---------------------------------------------------------------------------
# Campo de temperatura para el gradiente (parte 3)
# ---------------------------------------------------------------------------

def temperature_field(pos_history, vel_history, L, nx, ny, mass=1.0,
                      skip_first=0):
    """
    Estima el campo de temperatura local T(x, y) promediando la energía
    cinética en celdas de una malla nx × ny.

    En cada celda de área δΣ = (L/nx) × (L/ny) y para cada snapshot se
    acumula la energía cinética de las partículas presentes en la celda.
    La temperatura local se obtiene como:

        T_local(celda) = 2·<KE_celda> / (kB · <n_celda> · d)

    con d = 2 grados de libertad por partícula en 2D y kB = 1.

    Parameters
    ----------
    pos_history : (M, N, 2) — posiciones en M snapshots.
    vel_history : (M, N, 2) — velocidades en M snapshots.
    L, nx, ny   : parámetros de la malla.
    skip_first  : int — ignorar los primeros skip_first snapshots.

    Returns
    -------
    T_map   : (nx, ny) ndarray — temperatura media en cada celda.
    x_edges : (nx+1,) — bordes de la malla en x.
    y_edges : (ny+1,) — bordes de la malla en y.
    """
    pos_arr = np.asarray(pos_history)[skip_first:]
    vel_arr = np.asarray(vel_history)[skip_first:]

    M = pos_arr.shape[0]

    x_edges = np.linspace(0, L, nx + 1)
    y_edges = np.linspace(0, L, ny + 1)

    KE_sum = np.zeros((nx, ny))
    N_sum  = np.zeros((nx, ny))

    for t in range(M):
        pos_t = pos_arr[t]  # (N, 2)
        vel_t = vel_arr[t]

        ix = np.searchsorted(x_edges, pos_t[:, 0], side='right') - 1
        iy = np.searchsorted(y_edges, pos_t[:, 1], side='right') - 1

        # Clip: partículas en el borde exacto
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)

        ke = 0.5 * mass * (vel_t[:, 0]**2 + vel_t[:, 1]**2)

        for i in range(len(pos_t)):
            KE_sum[ix[i], iy[i]] += ke[i]
            N_sum[ix[i], iy[i]]  += 1.0

    # Temperatura local: T = 2<KE> / (d · <N>),  d=2
    with np.errstate(invalid='ignore'):
        T_map = np.where(N_sum > 0, KE_sum / N_sum, np.nan)

    return T_map, x_edges, y_edges


def plot_temperature_field(T_map, x_edges, y_edges, ax=None, cmap='inferno',
                           title='Campo de temperatura T(x,y)'):
    """Dibuja el mapa de temperaturas."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(xc, yc, indexing='ij')

    im = ax.pcolormesh(X, Y, T_map, cmap=cmap, shading='auto')
    fig.colorbar(im, ax=ax, label=r'$T$ (unid. reducidas)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')
    fig.tight_layout()
    return fig
