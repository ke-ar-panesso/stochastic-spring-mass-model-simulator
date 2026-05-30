# Animación del sistema masa-resorte para el oscilador armónico amortiguado

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

COLORS = {
    'bg':         '#0d1117',
    'panel':      '#161b22',
    'wall':       '#484f58',
    'spring':     '#58a6ff',
    'mass':       '#f78166',
    'mass_edge':  '#ffa657',
    'ground':     '#30363d',
    'text':       '#c9d1d9',
    'accent':     '#7c6aef',
    'trace_x':    '#58a6ff',
    'trace_v':    '#f778ba',
    'grid':       '#21262d',
    'eq_line':    '#3fb950',
    'mean_line':  '#ffd700',
}


def _spring_path(x_start, x_end, y_center=0.0, n_coils=14, amplitude=0.13):
    length = x_end - x_start
    if length < 0.05:
        return np.array([x_start, x_end]), np.array([y_center, y_center])

    lead = max(length * 0.07, 0.02)
    coil_start = x_start + lead
    coil_end   = x_end - lead

    if coil_end <= coil_start:
        return np.array([x_start, x_end]), np.array([y_center, y_center])

    n_pts = 2 * n_coils + 1
    zx = np.linspace(coil_start, coil_end, n_pts)
    zy = np.zeros(n_pts)
    for i in range(1, n_pts - 1):
        zy[i] = y_center + amplitude * ((-1) ** (i + 1))
    zy[0]  = y_center
    zy[-1] = y_center

    xs = np.concatenate([[x_start], zx, [x_end]])
    ys = np.concatenate([[y_center], zy, [y_center]])
    return xs, ys


def create_spring_animation(t, x, v, params):
    target_fps = 30
    max_duration = 20                       
    max_frames = target_fps * max_duration   
    total_steps = len(t) - 1
    skip = max(1, total_steps // max_frames)
    frame_indices = list(range(0, total_steps + 1, skip))
    interval_ms = 1000 // target_fps        

    with plt.rc_context({
        'figure.facecolor': COLORS['bg'],
        'axes.facecolor':   COLORS['bg'],
        'axes.edgecolor':   COLORS['grid'],
        'axes.labelcolor':  COLORS['text'],
        'text.color':       COLORS['text'],
        'xtick.color':      COLORS['text'],
        'ytick.color':      COLORS['text'],
        'grid.color':       COLORS['grid'],
        'grid.alpha':       0.3,
        'font.family':      'sans-serif',
    }):
        fig = plt.figure(figsize=(12, 8))
        fig.canvas.manager.set_window_title(
            'Animación — Oscilador Armónico Estocástico')

        gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.32,
                              left=0.08, right=0.94, top=0.92, bottom=0.08)
        ax_sys   = fig.add_subplot(gs[0])   
        ax_trace = fig.add_subplot(gs[1])

        x_range = max(abs(x.max()), abs(x.min()), 0.8) * 1.3
        wall_x  = -(x_range + 0.8)
        mass_w  = 0.45
        mass_h  = 0.50

        ax_sys.set_xlim(wall_x - 0.6, x_range + 1.0)
        ax_sys.set_ylim(-1.0, 1.3)
        ax_sys.set_aspect('equal')
        ax_sys.axis('off')
        ax_sys.set_title(
            'Oscilador Armónico Amortiguado — Simulación en Tiempo Real',
            fontsize=13, fontweight='bold', color=COLORS['accent'], pad=12)

        wall_w = 0.25
        wall = patches.Rectangle(
            (wall_x - wall_w, -0.55), wall_w, 1.1,
            facecolor=COLORS['wall'], edgecolor=COLORS['text'],
            linewidth=1.5, zorder=3)
        ax_sys.add_patch(wall)

        for yh in np.linspace(-0.45, 0.45, 7):
            ax_sys.plot([wall_x - wall_w, wall_x],
                        [yh + 0.08, yh - 0.08],
                        color=COLORS['text'], alpha=0.35, lw=0.8)

        ground_y = -mass_h / 2 - 0.05
        ax_sys.plot([wall_x - 0.6, x_range + 1.0], [ground_y, ground_y],
                    color=COLORS['ground'], lw=2.5)
        for xg in np.arange(wall_x - 0.4, x_range + 1.0, 0.22):
            ax_sys.plot([xg, xg - 0.12], [ground_y, ground_y - 0.13],
                        color=COLORS['ground'], lw=0.8, alpha=0.55)

        ax_sys.axvline(0, color=COLORS['eq_line'], ls='--', alpha=0.25, lw=1)
        ax_sys.text(0, ground_y - 0.22, 'x = 0', ha='center', fontsize=8,
                    color=COLORS['eq_line'], alpha=0.55)

        spring_line, = ax_sys.plot([], [], color=COLORS['spring'], lw=2.2,
                                   solid_capstyle='round', zorder=4)

        mass_rect = patches.FancyBboxPatch(
            (x[0] - mass_w / 2, -mass_h / 2), mass_w, mass_h,
            boxstyle="round,pad=0.04",
            facecolor=COLORS['mass'], edgecolor=COLORS['mass_edge'],
            linewidth=2.2, zorder=5)
        ax_sys.add_patch(mass_rect)

        mass_label = ax_sys.text(
            x[0], 0, 'm', ha='center', va='center',
            fontsize=13, fontweight='bold', color='white', zorder=6)

        info_text = ax_sys.text(
            0.99, 0.97, '', transform=ax_sys.transAxes,
            fontsize=10, va='top', ha='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.55', facecolor=COLORS['panel'],
                      edgecolor=COLORS['accent'], alpha=0.92, lw=1.5),
            zorder=10)

        ax_trace.set_xlim(0, t[-1])
        pad_x = (x.max() - x.min()) * 0.15 + 0.1
        ax_trace.set_ylim(x.min() - pad_x, x.max() + pad_x)
        ax_trace.set_xlabel('Tiempo (s)', fontsize=11)
        ax_trace.set_ylabel('Posición X(t)', fontsize=11,
                            color=COLORS['trace_x'])
        ax_trace.tick_params(axis='y', colors=COLORS['trace_x'])
        ax_trace.grid(True, alpha=0.18)

        ax_v = ax_trace.twinx()
        pad_v = (v.max() - v.min()) * 0.15 + 0.1
        ax_v.set_ylim(v.min() - pad_v, v.max() + pad_v)
        ax_v.set_ylabel('Velocidad V(t)', fontsize=11,
                        color=COLORS['trace_v'])
        ax_v.tick_params(axis='y', colors=COLORS['trace_v'])
        ax_v.spines['right'].set_color(COLORS['trace_v'])
        ax_v.spines['left'].set_color(COLORS['trace_x'])

        line_x, = ax_trace.plot([], [], color=COLORS['trace_x'], lw=1.6,
                                alpha=0.9, label='X(t)')
        line_v, = ax_v.plot([], [], color=COLORS['trace_v'], lw=1.2,
                            alpha=0.75, label='V(t)')
        dot_x, = ax_trace.plot([], [], 'o', color=COLORS['trace_x'],
                               ms=5, zorder=5)
        dot_v, = ax_v.plot([], [], 'o', color=COLORS['trace_v'],
                           ms=5, zorder=5)

        lines  = [line_x, line_v]
        labels = ['Posición X(t)', 'Velocidad V(t)']
        ax_trace.legend(lines, labels, loc='upper right', fontsize=9,
                        facecolor=COLORS['panel'], edgecolor=COLORS['grid'],
                        labelcolor=COLORS['text'])

        def init():
            spring_line.set_data([], [])
            line_x.set_data([], [])
            line_v.set_data([], [])
            dot_x.set_data([], [])
            dot_v.set_data([], [])
            info_text.set_text('')
            return (spring_line, mass_rect, mass_label,
                    line_x, line_v, dot_x, dot_v, info_text)

        def update(frame_num):
            i = frame_indices[frame_num]
            xi, vi, ti = x[i], v[i], t[i]

            sx, sy = _spring_path(wall_x, xi - mass_w / 2)
            spring_line.set_data(sx, sy)

            mass_rect.set_x(xi - mass_w / 2)
            mass_rect.set_y(-mass_h / 2)
            mass_label.set_position((xi, 0))

            nt = params.get('noise_type', 'standard')
            if nt == 'fractional':
                noise_lbl = f"fBm  H = {params.get('H', 0.7):.2f}"
            else:
                noise_lbl = "Browniano Estándar"
            info_text.set_text(
                f"  t   = {ti:8.3f} s\n"
                f"  X(t)= {xi:+8.4f}\n"
                f"  V(t)= {vi:+8.4f}\n"
                f"  Ruido: {noise_lbl}")

            idx = i + 1
            line_x.set_data(t[:idx], x[:idx])
            line_v.set_data(t[:idx], v[:idx])
            dot_x.set_data([ti], [xi])
            dot_v.set_data([ti], [vi])

            return (spring_line, mass_rect, mass_label,
                    line_x, line_v, dot_x, dot_v, info_text)

        anim = FuncAnimation(
            fig, update,
            frames=len(frame_indices),
            init_func=init,
            interval=interval_ms,
            blit=False,
            repeat=False)

    return fig, anim
