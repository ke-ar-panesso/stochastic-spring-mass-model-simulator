# Simulación de Trayectorias para el Oscilador Armónico

import numpy as np
from scipy.linalg import expm, toeplitz, cholesky

def generate_noise_increments(n, dt):
    # Browniano estándar: ΔB_i = √Δt · N(0,1)
    return np.sqrt(dt) * np.random.randn(n)

def simulate_trajectory(m, k, gamma, sigma, x0, v0, dt, t_final):
    """
    Ecuaciones de Euler-Maruyama:
        X_{i} = X_{i-1} + V_{i-1} · Δt
        V_{i} = V_{i-1} + (-γ/m · V_{i-1} - k/m · X_{i-1}) · Δt + (σ/m) · ΔB_i
    """
    n_steps = int(round(t_final / dt))
    t = np.linspace(0, t_final, n_steps + 1)
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    x[0] = x0
    v[0] = v0

    # Incrementos de ruido
    dB = generate_noise_increments(n_steps, dt)

    for i in range(n_steps):
        x[i + 1] = x[i] + v[i] * dt
        v[i + 1] = (v[i] + (-gamma / m * v[i] - k / m * x[i]) * dt + (sigma / m) * dB[i])

    # Retornar vectores de tiempo, posición y velocidad
    return t, x, v


def simulate_multiple(m, k, gamma, sigma, x0, v0, dt, t_final, n_traj, callback=None):
    n_steps = int(round(t_final / dt))
    all_x = np.zeros((n_traj, n_steps + 1))
    all_v = np.zeros((n_traj, n_steps + 1))

    for j in range(n_traj):
        t, xj, vj = simulate_trajectory(
            m, k, gamma, sigma, x0, v0, dt, t_final
        )
        all_x[j] = xj
        all_v[j] = vj
        if callback:
            callback(j, n_traj)

    return t, all_x, all_v

def analytical_mean(t_array, m, k, gamma, x0, v0):
    A = np.array([[0.0, 1.0], [-k / m, -gamma / m]])
    Z0 = np.array([x0, v0])

    n = len(t_array)
    mean_x = np.zeros(n)
    mean_v = np.zeros(n)

    for i, t in enumerate(t_array):
        eAt = expm(A * t)
        Z = eAt @ Z0
        mean_x[i] = Z[0]
        mean_v[i] = Z[1]

    return mean_x, mean_v
