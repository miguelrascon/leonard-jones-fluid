"""
Clase principal de simulación MD 2D para un fluido de Lennard-Jones.

Encapsula el estado del sistema y el bucle de integración.  El diseño
es intencionalmente modular: se pueden cambiar las condiciones de
contorno, el termostato o el integrador sin tocar la lógica de la clase.
"""

import numpy as np
import time

from .forces      import compute_forces
from .integrators import velocity_verlet_step
from .boundaries  import apply_walls, WallConfig
from .thermostat  import berendsen_rescale
from .analysis    import instant_temperature, remove_cm_velocity


def _initial_positions_grid(N, L, margin=0.6):
    """
    Coloca las partículas en una rejilla cuadrada con pequeña perturbación.
    El margen evita que las partículas comiencen muy cerca de las paredes.
    """
    n_side = int(np.ceil(np.sqrt(N)))
    xs = np.linspace(margin, L - margin, n_side)
    ys = np.linspace(margin, L - margin, n_side)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pos = np.column_stack([grid_x.ravel(), grid_y.ravel()])[:N]
    return pos


class Simulation:
    """
    Simulación MD de un fluido Lennard-Jones 2D en una caja cuadrada.

    Uso mínimo
    ----------
    >>> sim = Simulation(N=100, L=10.0, T_init=1.0)
    >>> sim.run(n_steps=10000, thermostat='berendsen', T_target=1.0)
    >>> print(sim.temperature_history[-1])

    Atributos
    ---------
    pos, vel : (N, 2) — estado actual.
    forces   : (N, 2) — fuerzas actuales.
    step     : int    — contador de pasos.
    temperature_history, energy_history, potential_history : listas.
    pos_history, vel_history : listas (guardadas cada n_save pasos).
    """

    def __init__(self, N, L, T_init=1.0, dt=0.005, r_cut=2.5,
                 mass=1.0, seed=None):
        """
        Parameters
        ----------
        N      : número de partículas.
        L      : lado de la caja cuadrada.
        T_init : temperatura de las condiciones iniciales.
        dt     : paso temporal.
        r_cut  : radio de corte del potencial LJ.
        mass   : masa de las partículas.
        seed   : semilla aleatoria.
        """
        self.N      = N
        self.L      = L
        self.dt     = dt
        self.r_cut  = r_cut
        self.mass   = mass
        self.step   = 0
        self.rng    = np.random.default_rng(seed)

        # Inicialización
        self.pos = _initial_positions_grid(N, L)
        self._init_velocities(T_init)
        self.forces, self._U = compute_forces(self.pos, r_cut)

        # Historiales
        self.temperature_history = []
        self.energy_history      = []   # energía total E = KE + PE
        self.potential_history   = []
        self.lambda_history      = []   # factor λ del termostato
        self.pos_history         = []
        self.vel_history         = []

        # Tiempo real de simulación
        self._t_wall = 0.0

    # ------------------------------------------------------------------
    # Inicialización de velocidades
    # ------------------------------------------------------------------

    def _init_velocities(self, T):
        """Maxwell-Boltzmann a temperatura T, sin velocidad de CM."""
        sigma = np.sqrt(T / self.mass)
        self.vel = self.rng.normal(0.0, sigma, size=(self.N, 2))
        self.vel = remove_cm_velocity(self.vel)

    # ------------------------------------------------------------------
    # Bucle de integración
    # ------------------------------------------------------------------

    def run(self, n_steps, wall_cfg=None, thermostat='none',
            T_target=1.0, tau_beren=0.1,
            n_save=50, verbose=True, verbose_interval=2000):
        """
        Ejecuta n_steps pasos de Velocity Verlet.

        Parameters
        ----------
        n_steps       : int — pasos a ejecutar.
        wall_cfg      : WallConfig o None.  Si None se usan paredes
                        elásticas por defecto.
        thermostat    : 'none', 'berendsen'.
        T_target      : float — temperatura objetivo (Berendsen).
        tau_beren     : float — tiempo de acoplamiento (Berendsen).
        n_save        : int — guardar snapshot cada n_save pasos.
        verbose       : bool — mostrar progreso.
        verbose_interval : int — pasos entre líneas de salida.
        """
        if wall_cfg is None:
            wall_cfg = WallConfig(self.L)   # todo elástico

        t0 = time.perf_counter()

        for _ in range(n_steps):
            # 1. Integrar
            self.pos, self.vel, self.forces, self._U = velocity_verlet_step(
                self.pos, self.vel, self.forces,
                self.dt, self.r_cut, self.mass
            )

            # 2. Condiciones de contorno
            apply_walls(self.pos, self.vel, wall_cfg, self.rng)

            # 3. Termostato
            lam = 1.0
            if thermostat == 'berendsen':
                self.vel, lam = berendsen_rescale(
                    self.vel, T_target, self.dt, tau_beren, self.mass
                )

            self.step += 1

            # 4. Observables
            T_inst = instant_temperature(self.vel, self.mass)
            KE     = 0.5 * self.mass * np.sum(self.vel**2)
            E      = KE + self._U

            self.temperature_history.append(T_inst)
            self.potential_history.append(self._U)
            self.energy_history.append(E)
            self.lambda_history.append(lam)

            if self.step % n_save == 0:
                self.pos_history.append(self.pos.copy())
                self.vel_history.append(self.vel.copy())

            if verbose and self.step % verbose_interval == 0:
                elapsed = time.perf_counter() - t0
                print(f"  paso {self.step:7d}  |  T={T_inst:.4f}  "
                      f"E={E:.4f}  KE={KE:.4f}  U={self._U:.4f}  "
                      f"[{elapsed:.1f}s]")

        self._t_wall += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def reset_histories(self):
        """Limpia los historiales (útil entre equilibración y producción)."""
        self.temperature_history.clear()
        self.energy_history.clear()
        self.potential_history.clear()
        self.lambda_history.clear()
        self.pos_history.clear()
        self.vel_history.clear()

    def summary(self):
        """Imprime un resumen estadístico de la producción."""
        T_arr = np.array(self.temperature_history)
        E_arr = np.array(self.energy_history)
        U_arr = np.array(self.potential_history)
        print(f"\n{'─'*52}")
        print(f"  Resumen de la simulación  (N={self.N}, L={self.L})")
        print(f"{'─'*52}")
        print(f"  Pasos totales   : {self.step}")
        print(f"  T medio         : {T_arr.mean():.4f}  ±  {T_arr.std():.4f}")
        print(f"  E total media   : {E_arr.mean():.4f}  ±  {E_arr.std():.4f}")
        print(f"  U/N media       : {U_arr.mean()/self.N:.4f}  ±  {U_arr.std()/self.N:.4f}")
        print(f"  Tiempo de pared : {self._t_wall:.1f} s")
        print(f"{'─'*52}\n")

    def snapshot_plot(self, ax=None, color='steelblue', alpha=0.8,
                      size=12, title=''):
        """Dibuja la posición actual de las partículas."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(self.pos[:, 0], self.pos[:, 1],
                   s=size, c=color, alpha=alpha)
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.set_aspect('equal')
        ax.set_title(title or f'Snapshot  paso={self.step}')
        return ax

    def plot_histories(self, ax=None):
        """Evolución temporal de T y E."""
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        steps = np.arange(1, len(self.temperature_history) + 1)
        ax[0].plot(steps, self.temperature_history, lw=0.6, alpha=0.8)
        ax[0].set_ylabel('T (unid. reducidas)')
        ax[0].set_title('Temperatura instantánea')
        ax[1].plot(steps, self.energy_history, lw=0.6, alpha=0.8)
        ax[1].set_ylabel('E total')
        ax[1].set_xlabel('paso')
        ax[1].set_title('Energía total')
        plt.tight_layout()
        return ax
