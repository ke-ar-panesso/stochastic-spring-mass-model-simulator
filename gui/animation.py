import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D         
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.rcParams['toolbar'] = 'None'

BG       = '#f6f8fa'
PANEL    = '#ffffff'
SPRING   = '#79c0ff'
ACCENT   = '#388bfd'
TXT      = '#24292f'
ANCHOR   = '#586069'
EQ_C     = '#3fb950'

_FACE_COLS = ['#ff9f97',   
              '#7a1e18',   
              '#ff7b72',   
              '#c24b43',   
              '#d96058',   
              '#f08878']   
_EDGE_C    = '#cc3d33'

def _spring_helix(z0, z1, radius=0.13, n_coils=12, n_pts=120):
    if abs(z1 - z0) < 0.03:
        zl = np.linspace(z0, z1, 5)
        return np.zeros(5), np.zeros(5), zl
    sign = np.sign(z1 - z0)
    lead = abs(z1 - z0) * 0.055
    zs, ze = z0 + sign * lead, z1 - sign * lead
    theta = np.linspace(0, 2 * np.pi * n_coils, n_pts)
    xh = radius * np.cos(theta)
    yh = radius * np.sin(theta)
    zh = np.linspace(zs, ze, n_pts)
    return np.r_[0, xh, 0], np.r_[0, yh, 0], np.r_[z0, zh, z1]


def _cube_faces(cx, cy, cz, s):
    r = s / 2
    return [
        [[cx-r,cy-r,cz+r],[cx+r,cy-r,cz+r],[cx+r,cy+r,cz+r],[cx-r,cy+r,cz+r]],
        [[cx-r,cy-r,cz-r],[cx+r,cy-r,cz-r],[cx+r,cy+r,cz-r],[cx-r,cy+r,cz-r]],
        [[cx-r,cy-r,cz-r],[cx+r,cy-r,cz-r],[cx+r,cy-r,cz+r],[cx-r,cy-r,cz+r]],
        [[cx-r,cy+r,cz-r],[cx+r,cy+r,cz-r],[cx+r,cy+r,cz+r],[cx-r,cy+r,cz+r]],
        [[cx-r,cy-r,cz-r],[cx-r,cy+r,cz-r],[cx-r,cy+r,cz+r],[cx-r,cy-r,cz+r]],
        [[cx+r,cy-r,cz-r],[cx+r,cy+r,cz-r],[cx+r,cy+r,cz+r],[cx+r,cy-r,cz+r]],
    ]


def _disc_surface(cx, cy, cz, r, n_r=5, n_th=40):
    th = np.linspace(0, 2 * np.pi, n_th)
    radii = np.linspace(0, r, n_r)
    T, R = np.meshgrid(th, radii)
    return R * np.cos(T) + cx, R * np.sin(T) + cy, np.full_like(T, cz)

def create_spring_animation(t, x, v, params):

    target_fps  = 30
    max_frames  = target_fps * 20
    total_steps = len(t) - 1
    skip        = max(1, total_steps // max_frames)
    frames_idx  = list(range(0, total_steps + 1, skip))
    interval_ms = 1000 // target_fps

    cube_s    = 0.7
    half_s    = cube_s / 2
    ceiling_z = 3.0
    x_amp     = max(float(np.abs(x).max()), 0.8)
    lim       = 1.0
    z_bot     = -(x_amp * 1.45 + 0.5)

    fig = plt.figure(figsize=(6.5, 7.5), facecolor=BG)
    fig.canvas.toolbar_visible = False
    
    fig.canvas.manager.set_window_title(
        'Animación 3D — Oscilador Armónico Estocástico')

    fig.text(0.5, 0.985, 'Oscilador Armónico Amortiguado Estocástico',
             ha='center', va='top', fontsize=14, fontweight='bold',
             color=ACCENT, fontfamily='DejaVu Sans')
             
    # Subtítulo (bajamos de 0.952 a 0.940)
    fig.text(0.5, 0.940, 'Euler-Maruyama  ·  Trayectoria individual',
             ha='center', va='top', fontsize=9,
             color='#8b949e', fontfamily='DejaVu Sans')

    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(top=0.88, bottom=0.0, left=0.0, right=1.0)

    ax.set_facecolor(BG)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor((0, 0, 0, 0))
    ax.set_axis_off()
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(z_bot - 0.1, ceiling_z + 0.35)
    ax.view_init(elev=5, azim=-90)
    ax.set_proj_type('persp', focal_length=0.2)
    ax.disable_mouse_rotation()      

    dx, dy, dz = _disc_surface(0, 0, ceiling_z, 0.24)
    ax.plot_surface(dx, dy, dz, color=ANCHOR, shade=True, alpha=1.0)
    th = np.linspace(0, 2 * np.pi, 48)
    ax.plot(0.24*np.cos(th), 0.24*np.sin(th),
            np.full(48, ceiling_z), color='#8b949e', lw=1.5, alpha=0.85)

    for seg in [([-.5, .5],[0,0],[0,0]), ([0,0],[-.5,.5],[0,0])]:
        ax.plot(*seg, '--', color=EQ_C, alpha=0.38, lw=1.0)
    ax.text(0.53, 0, 0.03, 'z = 0', color=EQ_C, alpha=0.6, fontsize=7.5)

    sx0, sy0, sz0 = _spring_helix(ceiling_z, float(x[0]) + half_s)
    spring_line, = ax.plot(sx0, sy0, sz0,
                           color=SPRING, lw=2.2,
                           solid_capstyle='round', alpha=0.95)

    init_verts = _cube_faces(0, 0, float(x[0]), cube_s)
    mass_cube  = Poly3DCollection(
        init_verts,
        facecolors=_FACE_COLS,
        edgecolors=_EDGE_C,
        linewidths=1.2,
        alpha=1.0)
    ax.add_collection3d(mass_cube)

    info = fig.text(
        0.5, 0.895, '',
        fontsize=10.5, va='top', ha='center', fontfamily='Consolas',
        color=TXT,
        bbox=dict(boxstyle='round,pad=0.55', facecolor=PANEL,
                  edgecolor=ACCENT, alpha=0.93, lw=1.6))

    def _refresh(fi):
        zi = float(x[fi])
        vi = float(v[fi])
        ti = float(t[fi])

        sx, sy, sz = _spring_helix(ceiling_z, zi + half_s)
        spring_line.set_data_3d(sx, sy, sz)

        mass_cube.set_verts(_cube_faces(0, 0, zi, cube_s))

        info.set_text(
            f"  t = {ti:7.3f} s   │   X(t) = {zi:+8.4f}"
            f"   │   V(t) = {vi:+8.4f}   │   σ = {params['sigma']:.3f}  ")

    def init():
        _refresh(frames_idx[0])
        return spring_line, mass_cube, info

    def update(frame_num):
        _refresh(frames_idx[frame_num])
        return spring_line, mass_cube, info

    anim = FuncAnimation(
        fig, update,
        frames=len(frames_idx),
        init_func=init,
        interval=interval_ms,
        blit=False,
        repeat=False)

    return fig, anim