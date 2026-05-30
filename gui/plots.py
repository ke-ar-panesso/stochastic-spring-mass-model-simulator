# Gráficas de trayectorias y valores medios para el oscilador armónico

import numpy as np
import matplotlib.pyplot as plt

COLORS = {
    'bg':        '#0d1117',
    'panel':     '#161b22',
    'grid':      '#21262d',
    'text':      '#c9d1d9',
    'accent':    '#7c6aef',
    'traj_x':    '#58a6ff',
    'traj_v':    '#f778ba',
    'mean_emp':  '#3fb950',
    'mean_anal': '#ffd700',
}


def create_trajectory_plots(t, all_x, all_v, mean_x_anal, mean_v_anal, params):
    n_traj = all_x.shape[0]

    mean_x_emp = np.mean(all_x, axis=0)
    mean_v_emp = np.mean(all_v, axis=0)

    with plt.rc_context({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor':   COLORS['bg'],
        'axes.edgecolor':   COLORS['grid'],
        'axes.labelcolor':  COLORS['text'],
        'text.color':       COLORS['text'],
        'xtick.color':      COLORS['text'],
        'ytick.color':      COLORS['text'],
        'grid.color':       COLORS['grid'],
        'grid.alpha':       0.25,
        'font.family':      'sans-serif',
    }):
        fig, (ax_x, ax_v) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={'hspace': 0.25, 'left': 0.09, 'right': 0.95,
                         'top': 0.91, 'bottom': 0.08})

        fig.canvas.manager.set_window_title(
            'Trayectorias — Oscilador Armónico Estocástico')

        nt = params.get('noise_type', 'standard')
        if nt == 'fractional':
            noise_str = f"Browniano Fraccionario (H={params.get('H', 0.7):.2f})"
        else:
            noise_str = "Browniano Estándar"

        fig.suptitle(
            f"Trayectorias del Oscilador  —  {n_traj} trayectorias  —  "
            f"Ruido: {noise_str}",
            fontsize=13, fontweight='bold', color=COLORS['accent'], y=0.97)

        alpha_traj = max(0.04, min(0.35, 5.0 / n_traj))

        for j in range(n_traj):
            ax_x.plot(t, all_x[j], color=COLORS['traj_x'],
                      alpha=alpha_traj, lw=0.7)

        ax_x.plot(t, mean_x_emp, color=COLORS['mean_emp'], lw=2.2,
                  label='Media empírica', zorder=4)
        ax_x.plot(t, mean_x_anal, color=COLORS['mean_anal'], lw=2.2,
                  ls='--', label='Media analítica  E[X(t)]', zorder=5)
        ax_x.set_ylabel('Posición  X(t)', fontsize=12)
        ax_x.grid(True, alpha=0.18)
        ax_x.legend(loc='upper right', fontsize=9,
                    facecolor=COLORS['panel'], edgecolor=COLORS['grid'],
                    labelcolor=COLORS['text'])
        ax_x.set_title('Posición', fontsize=11, color=COLORS['traj_x'],
                       loc='left', pad=6)

        for j in range(n_traj):
            ax_v.plot(t, all_v[j], color=COLORS['traj_v'],
                      alpha=alpha_traj, lw=0.7)

        ax_v.plot(t, mean_v_emp, color=COLORS['mean_emp'], lw=2.2,
                  label='Media empírica', zorder=4)
        ax_v.plot(t, mean_v_anal, color=COLORS['mean_anal'], lw=2.2,
                  ls='--', label='Media analítica  E[V(t)]', zorder=5)
        ax_v.set_ylabel('Velocidad  V(t)', fontsize=12)
        ax_v.set_xlabel('Tiempo  (s)', fontsize=12)
        ax_v.grid(True, alpha=0.18)
        ax_v.legend(loc='upper right', fontsize=9,
                    facecolor=COLORS['panel'], edgecolor=COLORS['grid'],
                    labelcolor=COLORS['text'])
        ax_v.set_title('Velocidad', fontsize=11, color=COLORS['traj_v'],
                       loc='left', pad=6)

    return fig
