"""
Parámetros del sistema para la simulación MD 2D de Lennard-Jones.
Unidades reducidas: sigma=1, epsilon=1, m=1, kB=1.
"""

# --- Sistema ---
N      = 100       # número de partículas
L      = 10.0      # lado de la caja cuadrada
RHO    = N / L**2  # densidad numérica (~0.1, fase gas/líquido diluido)

# --- Potencial LJ ---
SIGMA   = 1.0
EPSILON = 1.0
MASS    = 1.0
R_CUT   = 2.5      # radio de corte (en unidades de sigma)

# --- Integrador ---
DT       = 0.005   # paso temporal
N_EQUIL  = 30_000  # pasos de equilibración
N_PROD   = 80_000  # pasos de producción
N_SAVE   = 50      # guardar cada N_SAVE pasos

# --- Termostato de Berendsen ---
T_TARGET     = 1.0   # temperatura objetivo
TAU_BEREN    = 0.1   # tiempo de acoplamiento

# --- Semilla aleatoria ---
SEED = 1234

# --- Malla para el campo de temperatura (parte 3) ---
N_CELLS_X = 10
N_CELLS_Y = 10
