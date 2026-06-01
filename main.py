import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from core.simulator import simulate_multiple, analytical_mean
from gui.animation import create_spring_animation
from gui.plots import create_trajectory_plots

BG          = '#12121f'
FRAME_BG    = '#1a1a30'
ENTRY_BG    = '#262645'
FG          = '#c8cad8'
ACCENT      = '#7c6aef'
ACCENT_DARK = '#6355cc'
BTN_FG      = '#ffffff'
WARN        = '#ffa657'
SUCCESS     = '#3fb950'

class SimulatorApp:
    DEFAULTS = {
        'm':       1.0,
        'k':       4.0,
        'gamma':   0.5,
        'sigma':   0.3,
        'x0':      1.0,
        'v0':      0.0,
        'dt':      0.01,
        't_final': 10.0,
        'n_traj':  50,
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Simulador — Oscilador Armónico Estocástico')
        self.root.geometry('500x820')
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        self.vars = {}

        self._setup_style()
        self._build_ui()

    def _setup_style(self):
        s = ttk.Style()
        s.theme_use('clam')

        s.configure('.', background=BG, foreground=FG,
                    font=('Segoe UI', 10))
        s.configure('TFrame', background=BG)

        s.configure('TLabelframe', background=FRAME_BG,
                    foreground=ACCENT, borderwidth=2, relief='groove')
        s.configure('TLabelframe.Label', background=BG, foreground=ACCENT,
                    font=('Segoe UI', 10, 'bold'))

        s.configure('TLabel', background=FRAME_BG, foreground=FG,
                    font=('Segoe UI', 10))
        s.configure('Desc.TLabel', background=BG, foreground=FG,
                    font=('Segoe UI', 8))

        s.configure('TEntry', fieldbackground=ENTRY_BG, foreground='#ffffff',
                    insertcolor='#ffffff', font=('Consolas', 11),
                    borderwidth=1, padding=4)

        s.configure('TRadiobutton', background=FRAME_BG, foreground=FG,
                    font=('Segoe UI', 10), indicatorcolor=ACCENT)
        s.map('TRadiobutton',
              background=[('active', FRAME_BG)],
              indicatorcolor=[('selected', ACCENT)])

        s.configure('Accent.TButton',
                    background=ACCENT, foreground=BTN_FG,
                    font=('Segoe UI', 14, 'bold'),
                    padding=(20, 14), borderwidth=0)
        s.map('Accent.TButton',
              background=[('active', ACCENT_DARK), ('pressed', '#5245b0'),
                          ('disabled', '#3a3a5a')])

        s.configure('Status.TLabel', background=BG, foreground=FG,
                    font=('Segoe UI', 10), anchor='center')

    def _build_ui(self):
        container = ttk.Frame(self.root)
        container.pack(fill='both', expand=True, padx=18, pady=10)

        title = tk.Label(
            container, text='Oscilador Armónico\nAmortiguado Estocástico',
            font=('Segoe UI', 18, 'bold'), fg=ACCENT, bg=BG, justify='center')
        title.pack(pady=(4, 2))

        subtitle = tk.Label(
            container,
            text='Simulación por Método de Euler-Maruyama',
            font=('Segoe UI', 9), fg='#8888aa', bg=BG)
        subtitle.pack(pady=(0, 10))

        frm_model = ttk.LabelFrame(container,
                                   text='  Parámetros del Modelo  ',
                                   padding=(12, 8))
        frm_model.pack(fill='x', pady=(0, 6))
        frm_model.columnconfigure(1, weight=1)

        self._row(frm_model, 'm',     'Masa  (m)',                   0)
        self._row(frm_model, 'k',     'Constante resorte  (k)',      1)
        self._row(frm_model, 'gamma', 'Amortiguamiento  (γ)',        2)
        self._row(frm_model, 'sigma', 'Magnitud del ruido  (σ)',     3)

        frm_ic = ttk.LabelFrame(container,
                                text='  Condiciones Iniciales  ',
                                padding=(12, 8))
        frm_ic.pack(fill='x', pady=(0, 6))
        frm_ic.columnconfigure(1, weight=1)

        self._row(frm_ic, 'x0', 'Posición inicial  (X₀)', 0)
        self._row(frm_ic, 'v0', 'Velocidad inicial  (V₀)', 1)

        frm_sim = ttk.LabelFrame(container,
                                 text='  Parámetros de Simulación  ',
                                 padding=(12, 8))
        frm_sim.pack(fill='x', pady=(0, 6))
        frm_sim.columnconfigure(1, weight=1)

        self._row(frm_sim, 'dt',      'Paso de tiempo  (Δt)',    0)
        self._row(frm_sim, 't_final', 'Tiempo final  (T)',       1)
        self._row(frm_sim, 'n_traj',  'Número de trayectorias',  2)

        self.btn_sim = ttk.Button(
            container, text='▶   S I M U L A R',
            style='Accent.TButton', command=self._on_simulate)
        self.btn_sim.pack(pady=(14, 6), ipadx=20)

        self.status_var = tk.StringVar(value='Listo.')
        self.lbl_status = ttk.Label(
            container, textvariable=self.status_var, style='Status.TLabel')
        self.lbl_status.pack(pady=(0, 4))

        self.progress = ttk.Progressbar(
            container, length=400, mode='determinate')
        self.progress.pack(pady=(0, 6))

    def _row(self, parent, key, label, row):
        """Crea una fila con etiqueta + entrada para un parámetro."""
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky='w', padx=(4, 8), pady=4)
        var = tk.StringVar(value=str(self.DEFAULTS[key]))
        entry = ttk.Entry(parent, textvariable=var, width=14, justify='center')
        entry.grid(row=row, column=1, sticky='e', padx=(8, 4), pady=4)
        self.vars[key] = var

    def _parse_params(self):
        try:
            p = {}
            for key, var in self.vars.items():
                val = var.get().strip()
                if key == 'n_traj':
                    p[key] = int(val)
                else:
                    p[key] = float(val)

            if p['m'] <= 0:
                raise ValueError('La masa (m) debe ser positiva.')
            if p['k'] < 0:
                raise ValueError('La constante k no puede ser negativa.')
            if p['gamma'] < 0:
                raise ValueError('El amortiguamiento γ no puede ser negativo.')
            if p['dt'] <= 0:
                raise ValueError('El paso Δt debe ser positivo.')
            if p['t_final'] <= 0:
                raise ValueError('El tiempo final debe ser positivo.')
            if p['n_traj'] < 1:
                raise ValueError('Se necesita al menos 1 trayectoria.')

            return p

        except ValueError as e:
            messagebox.showerror('Error de parámetros', str(e))
            return None

    def _on_simulate(self):
        params = self._parse_params()
        if params is None:
            return

        self.btn_sim.configure(state='disabled')
        self.progress['value'] = 0
        self.progress['maximum'] = params['n_traj']

        def progress_cb(j, total):
            self.progress['value'] = j + 1
            self.status_var.set(
                f"Simulando trayectoria {j + 1} / {total} …")
            self.root.update_idletasks()

        try:
            self.status_var.set('Simulando trayectorias…')
            self.root.update()

            t, all_x, all_v = simulate_multiple(
                m=params['m'], k=params['k'], gamma=params['gamma'],
                sigma=params['sigma'], x0=params['x0'], v0=params['v0'],
                dt=params['dt'], t_final=params['t_final'],
                n_traj=params['n_traj'],
                callback=progress_cb)

            self.status_var.set('Calculando valor medio analítico…')
            self.root.update()

            mean_x_anal, mean_v_anal = analytical_mean(
                t, params['m'], params['k'], params['gamma'],
                params['x0'], params['v0'])

            self.status_var.set('Generando visualización…')
            self.root.update()

            plt.close('all')

            fig_plots = create_trajectory_plots(
                t, all_x, all_v, mean_x_anal, mean_v_anal, params)

            fig_anim, anim = create_spring_animation(
                t, all_x[0], all_v[0], params)

            self.status_var.set('✓  ¡Simulación completada!')
            self.lbl_status.configure(foreground=SUCCESS)
            self.root.update()

            self.root.withdraw()
            plt.show()
            self.root.deiconify()

            self.lbl_status.configure(foreground=FG)
            self.status_var.set('Listo.')

        except Exception as e:
            messagebox.showerror('Error en simulación', str(e))
            self.status_var.set('Error.')
            self.lbl_status.configure(foreground=WARN)

        finally:
            self.btn_sim.configure(state='normal')
            self.progress['value'] = 0

    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    app = SimulatorApp()
    app.run()
