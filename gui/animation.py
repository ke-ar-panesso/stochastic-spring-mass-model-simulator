import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


def _spring_path_3d(z_start, z_end, radius=0.15, n_coils=14):
    """
    Genera coordenadas para un resorte 3D vertical.
    """
    length = z_end - z_start
    if abs(length) < 0.05:
        return np.zeros(2), np.zeros(2), np.array([z_start, z_end])
        
    lead = max(abs(length) * 0.07, 0.02)
    sign = np.sign(z_end - z_start)
    coil_start = z_start + sign * lead
    coil_end = z_end - sign * lead

    if abs(coil_end - coil_start) < 0.01:
        return np.zeros(2), np.zeros(2), np.array([z_start, z_end])

    theta = np.linspace(0, 2 * np.pi * n_coils, 150)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.linspace(coil_start, coil_end, 150)
    
    xs = np.concatenate(([0], x, [0]))
    ys = np.concatenate(([0], y, [0]))
    zs = np.concatenate(([z_start], z, [z_end]))
    return xs, ys, zs


def _cube_faces(cx, cy, cz, size):
    """
    Genera las caras de un cubo 3D para usar con Poly3DCollection.
    """
    r = size / 2
    return [
        [[cx-r, cy-r, cz+r], [cx+r, cy-r, cz+r], [cx+r, cy+r, cz+r], [cx-r, cy+r, cz+r]], # Top
        [[cx-r, cy-r, cz-r], [cx+r, cy-r, cz-r], [cx+r, cy+r, cz-r], [cx-r, cy+r, cz-r]], # Bottom
        [[cx-r, cy-r, cz-r], [cx+r, cy-r, cz-r], [cx+r, cy-r, cz+r], [cx-r, cy-r, cz+r]], # Front
        [[cx-r, cy+r, cz-r], [cx+r, cy+r, cz-r], [cx+r, cy+r, cz+r], [cx-r, cy+r, cz+r]], # Back
        [[cx-r, cy-r, cz-r], [cx-r, cy+r, cz-r], [cx-r, cy+r, cz+r], [cx-r, cy-r, cz+r]], # Left
        [[cx+r, cy-r, cz-r], [cx+r, cy+r, cz-r], [cx+r, cy+r, cz+r], [cx+r, cy-r, cz+r]], # Right
    ]


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
        fig = plt.figure(figsize=(14, 7))
        fig.canvas.manager.set_window_title(
            'Animación 3D — Oscilador Armónico Estocástico')

        # Layout: 1 fila, 2 columnas. La izquierda toma más espacio.
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.25,
                              left=0.06, right=0.98, top=0.90, bottom=0.12)
        
        ax_trace = fig.add_subplot(gs[0])
        ax_sys = fig.add_subplot(gs[1], projection='3d')

        # ─── Configuración del sistema 3D ─────────────────────────────────────
        z_range = max(abs(x.max()), abs(x.min()), 0.8) * 1.3
        ceiling_z = z_range + 0.8
        mass_size = 0.7

        ax_sys.set_xlim(-0.6, 0.6)
        ax_sys.set_ylim(-0.6, 0.6)
        ax_sys.set_zlim(-(z_range + 0.5), ceiling_z + 0.1)
        
        # Eliminar fondo de los ejes 3D
        ax_sys.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_sys.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_sys.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_sys.grid(False)
        ax_sys.axis('off')
        
        ax_sys.set_title(
            'Oscilador Vertical 3D',
            fontsize=12, fontweight='bold', color=COLORS['accent'], pad=10)

        cx, cy, cz = _spring_path_3d(0, 0) # Just dummy variables
        w = 0.5
        # Techo dibujado como una superficie simple
        xx, yy = np.meshgrid([-w/2, w/2], [-w/2, w/2])
        ax_sys.plot_surface(xx, yy, np.full_like(xx, ceiling_z), color=COLORS['wall'], edgecolor=COLORS['wall'], lw=1.5, alpha=1.0)

        # Suelo
        floor_z = -(z_range + 0.5)
        ax_sys.plot_surface(xx, yy, np.full_like(xx, floor_z), color=COLORS['ground'], alpha=1.0)

        # Línea de equilibrio (z=0)
        ax_sys.plot([-w, w], [0, 0], [0, 0], color=COLORS['eq_line'], ls='--', alpha=0.4, lw=1)
        ax_sys.plot([0, 0], [-w, w], [0, 0], color=COLORS['eq_line'], ls='--', alpha=0.4, lw=1)
        ax_sys.text(0, 0, 0, 'z = 0', color=COLORS['eq_line'], alpha=0.7)

        # Elementos dinámicos del 3D
        spring_line, = ax_sys.plot([], [], [], color=COLORS['spring'], lw=3.0, solid_capstyle='round')
        
        faces = _cube_faces(0, 0, x[0], mass_size)
        mass_cube = Poly3DCollection(faces, facecolors=COLORS['mass'], edgecolors=COLORS['mass_edge'], alpha=1.0, linewidths=1.5)
        ax_sys.add_collection3d(mass_cube)
        
        # ─── Panel de información ─────────────────────────────────────────────
        info_text = fig.text(
            0.98, 0.95, '',
            fontsize=10, va='top', ha='right', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.55', facecolor=COLORS['panel'],
                      edgecolor=COLORS['accent'], alpha=0.92, lw=1.5),
            zorder=10)

        # ─── Configuración de trazas 2D ───────────────────────────────────────
        ax_trace.set_xlim(0, t[-1])
        pad_x = (x.max() - x.min()) * 0.15 + 0.1
        ax_trace.set_ylim(x.min() - pad_x, x.max() + pad_x)
        ax_trace.set_xlabel('Tiempo (s)', fontsize=11)
        ax_trace.set_ylabel('Posición Z(t)', fontsize=11, color=COLORS['trace_x'])
        ax_trace.tick_params(axis='y', colors=COLORS['trace_x'])
        ax_trace.grid(True, alpha=0.18)
        ax_trace.set_title('Evolución Temporal de Posición y Velocidad', 
                           fontsize=13, fontweight='bold', color=COLORS['accent'], loc='left', pad=12)

        ax_v = ax_trace.twinx()
        pad_v = (v.max() - v.min()) * 0.15 + 0.1
        ax_v.set_ylim(v.min() - pad_v, v.max() + pad_v)
        ax_v.set_ylabel('Velocidad V(t)', fontsize=11, color=COLORS['trace_v'])
        ax_v.tick_params(axis='y', colors=COLORS['trace_v'])
        ax_v.spines['right'].set_color(COLORS['trace_v'])
        ax_v.spines['left'].set_color(COLORS['trace_x'])

        line_x, = ax_trace.plot([], [], color=COLORS['trace_x'], lw=1.8, alpha=0.9, label='Posición Z(t)')
        line_v, = ax_v.plot([], [], color=COLORS['trace_v'], lw=1.5, alpha=0.75, label='Velocidad V(t)')
        dot_x, = ax_trace.plot([], [], 'o', color=COLORS['trace_x'], ms=6, zorder=5)
        dot_v, = ax_v.plot([], [], 'o', color=COLORS['trace_v'], ms=6, zorder=5)

        lines  = [line_x, line_v]
        labels = ['Posición Z(t)', 'Velocidad V(t)']
        ax_trace.legend(lines, labels, loc='upper right', fontsize=10,
                        facecolor=COLORS['panel'], edgecolor=COLORS['grid'],
                        labelcolor=COLORS['text'])

        def init():
            spring_line.set_data_3d([], [], [])
            line_x.set_data([], [])
            line_v.set_data([], [])
            dot_x.set_data([], [])
            dot_v.set_data([], [])
            info_text.set_text('')
            return spring_line, mass_cube, line_x, line_v, dot_x, dot_v, info_text

        def update(frame_num):
            i = frame_indices[frame_num]
            zi, vi, ti = x[i], v[i], t[i]

            # Actualizar resorte 3D (desde techo hasta masa)
            sx, sy, sz = _spring_path_3d(ceiling_z, zi + mass_size / 2)
            spring_line.set_data_3d(sx, sy, sz)

            # Actualizar masa 3D
            new_faces = _cube_faces(0, 0, zi, mass_size)
            mass_cube.set_verts(new_faces)

            info_text.set_text(
                f"  t   = {ti:8.3f} s\n"
                f"  Z(t)= {zi:+8.4f}\n"
                f"  V(t)= {vi:+8.4f}\n"
                f"  Ruido: Estándar")

            idx = i + 1
            line_x.set_data(t[:idx], x[:idx])
            line_v.set_data(t[:idx], v[:idx])
            dot_x.set_data([ti], [zi])
            dot_v.set_data([ti], [vi])

            return spring_line, mass_cube, line_x, line_v, dot_x, dot_v, info_text

        anim = FuncAnimation(
            fig, update,
            frames=len(frame_indices),
            init_func=init,
            interval=interval_ms,
            blit=False,
            repeat=False)

    return fig, anim

