# 2D Lennard-Jones Fluid Dynamics and Thermalization

## Overview 
This repository contains a Molecular Dynamics (MD) simulation of a 2D Lennard-Jones fluid consisting of $N$ particles within a square enclosure of area $L^2$. 

Instead of a standard constant-energy (NVE) simulation, this project explores different methods of temperature control and thermalization, forcing the average kinetic energy of the particles to a target temperature $k_B T$. The project analyzes both global thermostatting and local boundary thermalization, culminating in the study of non-equilibrium temperature gradients.

## Features & Project Phases

### 1. Global Temperature Control (Berendsen Thermostat)
The baseline simulation utilizes a **Berendsen thermostat** to globally scale particle velocities and maintain a constant temperature.
* Computes the steady-state dynamics of the fluid.
* Analyzes the linear momentum distribution of a randomly selected particle to verify alignment with the theoretical **Maxwell-Boltzmann distribution**.
* Evaluates the mean interaction energy (potential energy) per particle and its statistical dispersion.

### 2. Boundary Thermalization (Stochastic Thermal Walls)
To create a more physically realistic model, the global thermostat is replaced with localized boundary thermalization.
* Simulates a heat bath at temperature $T$ acting exclusively at the walls.
* Replaces standard elastic wall collisions with a stochastic process: upon impact, a particle reflects with a new random velocity drawn from the Maxwell-Boltzmann distribution for temperature $T$.
* Compares the thermodynamic results of this boundary-driven approach with the global Berendsen method.

### 3. Temperature Gradients (Non-Equilibrium Steady State)
The final phase models heat conduction by establishing mixed boundary conditions.
* Configures the left wall to temperature $T$ and the right wall to $T/2$.
* Keeps the top and bottom walls diathermal (perfectly elastic bounces).
* Calculates local temperatures by evaluating the kinetic energy density within small spatial grids of area $\Sigma$.
* Generates a **2D Heatmap** to visualize and analyze the resulting steady-state temperature gradient across the enclosure.

---
## Física implementada

Todo en **unidades
reducidas**: σ = ε = m = kB = 1.

### Potencial Lennard-Jones
```
V(r) = 4ε [(σ/r)^12 - (σ/r)^6]    desplazado en r_cut para continuidad
```

### Integrador: Velocity Verlet (simpléctico)
```
r(t+dt) = r(t) + v(t)·dt + ½a(t)·dt²
v(t+dt) = v(t) + ½[a(t) + a(t+dt)]·dt
```

### Termostato de Berendsen
```
v → λ·v,    λ = sqrt(1 + dt/τ · (T_ref/T_inst − 1))
```

### Paredes térmicas (distribución de flujo Maxwell-Boltzmann)
Al chocar con una pared a temperatura T:
- componente tangencial: v_t  ~ N(0, T)
- componente normal (hacia el interior): v_n ~ Rayleigh(√T)
  → muestreo: v_n = √(−2T·ln U),  U ~ Uniforme(0,1)

### Campo de temperatura T(x,y)
Se discretiza el recinto en celdas de área δΣ = (L/nx)·(L/ny) y se
promedia la energía cinética local sobre los snapshots de producción.

## Parámetros por defecto

| Parámetro | Valor |
|-----------|-------|
| N         | 100   |
| L         | 10.0  |
| ρ         | 0.10  |
| r_cut     | 2.5   |
| dt        | 0.005 |
| T_target  | 1.0   |
| τ_Beren   | 0.1   |


-- 


## Getting Started

### Prerequisites
*(List the libraries and tools required to run your code here. For example:)*
* Python 3.8+
* NumPy
* Matplotlib (for visualization)
* SciPy

### Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/yourusername/your-repo-name.git)

