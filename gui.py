"""
gui.py  –  Tkinter control panel for the drawing machine.
Optimised for 320 × 480 px (3.5" portrait touchscreen).

Two pages stacked in the same window:
  - Main page  : machine control
  - Config page: settings + serial console

Button states:
  - START / CANCEL : disabled until homed, toggles when job running
  - PAUSE / RESUME : disabled until homed + job running, resets on cancel
  - Jog controls   : disabled while job running (even if paused)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import importlib
import os
import sys

from machine import Machine, list_serial_ports
from config import GRBL_SETTINGS, PEN_DOWN_Z, JOG_STEP_MM

# ── palette ───────────────────────────────────────────────────────────────────
BG           = "#1a1a2e"
PANEL_BG     = "#16213e"
ACCENT       = "#0f3460"
BTN_START    = "#1a7a4a"
BTN_CANCEL   = "#8b1a1a"
BTN_PAUSE    = "#b5830a"
BTN_HOME     = "#1a4a7a"
BTN_JOG      = "#2a2a4a"
BTN_PEN_UP   = "#2a4a2a"
BTN_PEN_DN   = "#4a2a2a"
BTN_DISABLED = "#2a2a2a"
FG           = "#e0e0e0"
FG_DIM       = "#888888"
FG_DISABLED  = "#555555"
FG_STATUS    = "#00e5ff"
FG_CONSOLE   = "#00ff99"
FONT_LG      = ("Helvetica", 14, "bold")
FONT_MD      = ("Helvetica", 11)
FONT_SM      = ("Helvetica", 9)
FONT_MONO    = ("Courier", 10)
FONT_MONO_SM = ("Courier", 8)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def scan_models() -> list:
    if not os.path.isdir(MODELS_DIR):
        return []
    return [
        f[:-3]
        for f in sorted(os.listdir(MODELS_DIR))
        if f.endswith(".py") and not f.startswith("_")
    ]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Drawing Machine")
        self.attributes("-fullscreen", True)
        self.resizable(False, False)
        self.configure(bg=BG)

        self.machine = Machine()
        self._model_thread = None
        self._stop_event  = threading.Event()
        self._is_paused   = False
        self._pen_is_down = False
        self._job_running = False   # true between START and CANCEL
        self._homed       = False   # true after first successful home

        # ── page container ────────────────────────────────────────────────────
        self._container = tk.Frame(self, bg=BG)
        self._container.pack(fill="both", expand=True)
        self._container.grid_rowconfigure(0, weight=1)
        self._container.grid_columnconfigure(0, weight=1)

        self._main_page   = tk.Frame(self._container, bg=BG)
        self._config_page = tk.Frame(self._container, bg=BG)
        for page in (self._main_page, self._config_page):
            page.grid(row=0, column=0, sticky="nsew")

        self._build_main_page()
        self._build_config_page()
        self._show_main()

        self._refresh_ports()
        self._poll_status()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── page switching ─────────────────────────────────────────────────────────

    def _show_main(self):
        self._main_page.tkraise()

    def _show_config(self):
        self._config_page.tkraise()

    # ── MAIN PAGE ──────────────────────────────────────────────────────────────

    def _build_main_page(self):
        p = self._main_page

        # ── Connection bar + Model selector (combined grid) ───────────────────
        conn_frame = tk.Frame(p, bg=PANEL_BG)
        conn_frame.pack(fill="x", padx=4, pady=(4, 2))

        # Row 0: Port label | port combobox | ↺ | Connect | ✕
        tk.Label(conn_frame, text="Port", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SM).grid(row=0, column=0, padx=(6, 2), pady=4)

        self._port_var = tk.StringVar()
        self._port_cb = ttk.Combobox(
            conn_frame, textvariable=self._port_var,
            width=10, state="readonly", font=FONT_SM,
        )
        self._port_cb.grid(row=0, column=1, padx=2, pady=4)

        tk.Button(
            conn_frame, text="↺", bg=ACCENT, fg=FG,
            relief="flat", font=FONT_MD, width=2,
            command=self._refresh_ports, cursor="hand2",
        ).grid(row=0, column=2, padx=2)

        self._conn_btn = tk.Button(
            conn_frame, text="Connect", bg=BTN_HOME, fg=FG,
            relief="flat", font=FONT_SM, width=8, height=1,
            command=self._toggle_connect, cursor="hand2",
        )
        self._conn_btn.grid(row=0, column=3, padx=(2, 2), pady=4)

        tk.Button(
            conn_frame, text="✕", bg=BTN_CANCEL, fg=FG,
            relief="flat", font=FONT_MD, width=2,
            command=self._on_close, cursor="hand2",
        ).grid(row=0, column=4, padx=(2, 6), pady=4)

        # Row 1: Model label | model combobox | ↺ | ⚙
        tk.Label(conn_frame, text="Model", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SM).grid(row=1, column=0, padx=(6, 2), pady=4)

        self._model_var = tk.StringVar()
        self._model_cb = ttk.Combobox(
            conn_frame, textvariable=self._model_var,
            state="readonly", font=FONT_SM, width=10,
        )
        models = scan_models()
        self._model_cb["values"] = models
        if models:
            self._model_cb.current(0)
        self._model_cb.grid(row=1, column=1, padx=2, pady=4)

        tk.Button(
            conn_frame, text="↺", bg=ACCENT, fg=FG,
            relief="flat", font=FONT_MD, width=2,
            command=self._refresh_models, cursor="hand2",
        ).grid(row=1, column=2, padx=2)

        tk.Button(
            conn_frame, text="⚙", bg=ACCENT, fg=FG,
            relief="flat", font=FONT_MD, width=2,
            command=self._show_config, cursor="hand2",
        ).grid(row=1, column=3, padx=(2, 2), pady=4)

        # ── START/CANCEL + PAUSE/RESUME ───────────────────────────────────────
        ctrl_frame = tk.Frame(p, bg=BG)
        ctrl_frame.pack(fill="x", padx=4, pady=4)

        self._start_btn = tk.Button(
            ctrl_frame, text="▶  Start",
            command=self._on_start_cancel,
            bg=BTN_DISABLED, fg=FG_DISABLED,
            activebackground=BTN_DISABLED, activeforeground=FG_DISABLED,
            relief="flat", bd=0, highlightthickness=0,
            font=FONT_MD, height=2, cursor="hand2",
            state="disabled",
        )
        self._start_btn.pack(side="left", expand=True, fill="x", padx=2)

        self._pause_btn = tk.Button(
            ctrl_frame, text="⏸  Pause",
            command=self._on_pause_resume,
            bg=BTN_DISABLED, fg=FG_DISABLED,
            activebackground=BTN_DISABLED, activeforeground=FG_DISABLED,
            relief="flat", bd=0, highlightthickness=0,
            font=FONT_MD, height=2, cursor="hand2",
            state="disabled",
        )
        self._pause_btn.pack(side="left", expand=True, fill="x", padx=2)

        # ── Divider ───────────────────────────────────────────────────────────
        tk.Frame(p, bg=ACCENT, height=1).pack(fill="x", padx=4, pady=2)

        # ── Homing ────────────────────────────────────────────────────────────
        home_frame = tk.Frame(p, bg=PANEL_BG)
        home_frame.pack(fill="x", padx=4, pady=2)

        tk.Label(home_frame, text="HOMING", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SM).grid(row=0, column=0, columnspan=2,
                                    sticky="w", padx=6, pady=(4, 0))

        self._home_btn = tk.Button(
            home_frame, text="🏠 Home ($H)", command=self._on_home,
            bg=BTN_HOME, fg=FG, activebackground=BTN_HOME,
            relief="flat", font=FONT_SM, width=12, height=1, cursor="hand2",
        )
        self._home_btn.grid(row=1, column=0, padx=6, pady=4, sticky="ew")

        tk.Button(
            home_frame, text="Set Origin", command=self._on_set_origin,
            bg=ACCENT, fg=FG, activebackground=ACCENT,
            relief="flat", font=FONT_SM, width=10, height=1, cursor="hand2",
        ).grid(row=1, column=1, padx=6, pady=4, sticky="ew")

        # ── Jog + Pen ─────────────────────────────────────────────────────────
        jog_outer = tk.Frame(p, bg=PANEL_BG)
        jog_outer.pack(fill="x", padx=4, pady=2)

        settings_row = tk.Frame(jog_outer, bg=PANEL_BG)
        settings_row.pack(fill="x", padx=6, pady=(4, 2))

        tk.Label(settings_row, text="Jog:", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SM).pack(side="left")
        self._step_var = tk.StringVar(value=str(int(JOG_STEP_MM)))
        tk.Entry(settings_row, textvariable=self._step_var, width=4,
                 bg=ACCENT, fg=FG, insertbackground=FG, font=FONT_SM,
                 relief="flat").pack(side="left", padx=(2, 1))
        tk.Label(settings_row, text="mm", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SM).pack(side="left")

        tk.Label(settings_row, text="", bg=PANEL_BG, width=2).pack(side="left")

        tk.Label(settings_row, text="Z:", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SM).pack(side="left")
        self._z_var = tk.StringVar(value=str(int(PEN_DOWN_Z)))
        tk.Entry(settings_row, textvariable=self._z_var, width=4,
                 bg=ACCENT, fg=FG, insertbackground=FG, font=FONT_SM,
                 relief="flat").pack(side="left", padx=(2, 1))
        tk.Label(settings_row, text="mm", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SM).pack(side="left")
        tk.Button(settings_row, text="Set", command=self._on_set_z,
                  bg=ACCENT, fg=FG, relief="flat", font=FONT_SM,
                  width=3, cursor="hand2").pack(side="left", padx=(4, 0))

        # D-pad — keep refs so we can enable/disable
        jog_grid = tk.Frame(jog_outer, bg=PANEL_BG)
        jog_grid.pack(pady=4)

        btn_size = dict(width=4, height=2, font=FONT_LG)

        self._jog_btns = []

        def jog_btn(text, dx, dy, row, col):
            b = tk.Button(
                jog_grid, text=text,
                command=lambda: self._on_jog(dx, dy),
                bg=BTN_JOG, fg=FG, activebackground=BTN_JOG,
                relief="flat", bd=0, highlightthickness=0,
                cursor="hand2", **btn_size,
            )
            b.grid(row=row, column=col, padx=2, pady=2)
            self._jog_btns.append(b)
            return b

        jog_btn("↑",  0, -1, 0, 1)
        jog_btn("←", -1,  0, 1, 0)
        jog_btn("→",  1,  0, 1, 2)
        jog_btn("↓",  0,  1, 2, 1)

        self._pen_btn = tk.Button(
            jog_grid, text="✒\nUp",
            command=self._on_pen_toggle,
            bg=BTN_PEN_UP, fg=FG, activebackground=BTN_PEN_UP,
            relief="flat", bd=0, highlightthickness=0,
            font=("Helvetica", 9, "bold"), width=4, height=2, cursor="hand2",
        )
        self._pen_btn.grid(row=1, column=1, padx=2, pady=2)
        self._jog_btns.append(self._pen_btn)

        # ── Divider ───────────────────────────────────────────────────────────
        tk.Frame(p, bg=ACCENT, height=1).pack(fill="x", padx=4, pady=2)

        # ── Status bar ────────────────────────────────────────────────────────
        status_frame = tk.Frame(p, bg=PANEL_BG)
        status_frame.pack(fill="both", expand=True, padx=4, pady=(2, 4))

        self._status_lbl = tk.Label(
            status_frame, text="Status: Disconnected",
            bg=PANEL_BG, fg=FG_STATUS, font=FONT_MD, anchor="w",
        )
        self._status_lbl.pack(fill="x", padx=8, pady=(8, 4))

        coords_frame = tk.Frame(status_frame, bg=PANEL_BG)
        coords_frame.pack(fill="x", padx=8, pady=(0, 6))

        tk.Label(coords_frame, text="X", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SM).pack(side="left")
        self._x_lbl = tk.Label(coords_frame, text="—", bg=PANEL_BG,
                                fg=FG_STATUS, font=("Courier", 14, "bold"),
                                width=7, anchor="w")
        self._x_lbl.pack(side="left", padx=(3, 2))
        tk.Label(coords_frame, text="mm", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SM).pack(side="left", padx=(0, 12))

        tk.Label(coords_frame, text="Y", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SM).pack(side="left")
        self._y_lbl = tk.Label(coords_frame, text="—", bg=PANEL_BG,
                                fg=FG_STATUS, font=("Courier", 14, "bold"),
                                width=7, anchor="w")
        self._y_lbl.pack(side="left", padx=(3, 2))
        tk.Label(coords_frame, text="mm", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SM).pack(side="left")

    # ── CONFIG PAGE ────────────────────────────────────────────────────────────

    def _build_config_page(self):
        p = self._config_page

        header = tk.Frame(p, bg=PANEL_BG)
        header.pack(fill="x", padx=4, pady=(4, 2))

        tk.Button(
            header, text="←", bg=ACCENT, fg=FG,
            relief="flat", font=FONT_MD, width=2,
            command=self._show_main, cursor="hand2",
        ).pack(side="left", padx=(6, 8), pady=4)

        tk.Label(header, text="Config", bg=PANEL_BG, fg=FG,
                 font=FONT_MD).pack(side="left", pady=4)

        tk.Frame(p, bg=ACCENT, height=1).pack(fill="x", padx=4, pady=2)

        tk.Label(p, text="MACHINE SETTINGS", bg=BG, fg=FG_DIM,
                 font=FONT_SM).pack(anchor="w", padx=10, pady=(8, 2))

        settings_btns = tk.Frame(p, bg=BG)
        settings_btns.pack(fill="x", padx=8, pady=(0, 4))

        tk.Button(
            settings_btns, text="⬆  Push to Firmware",
            command=self._on_push_settings,
            bg=BTN_HOME, fg=FG, activebackground=BTN_HOME,
            relief="flat", font=FONT_SM, cursor="hand2", height=2,
        ).pack(side="left", expand=True, fill="x", padx=(0, 2))

        tk.Button(
            settings_btns, text="⬇  Download from Firmware",
            command=self._on_download_settings,
            bg=ACCENT, fg=FG, activebackground=ACCENT,
            relief="flat", font=FONT_SM, cursor="hand2", height=2,
        ).pack(side="left", expand=True, fill="x", padx=(2, 0))

        tk.Label(p, text="SERIAL CONSOLE", bg=BG, fg=FG_DIM,
                 font=FONT_SM).pack(anchor="w", padx=10, pady=(4, 2))

        # Input row anchored at bottom
        input_frame = tk.Frame(p, bg=PANEL_BG)
        input_frame.pack(fill="x", padx=4, pady=(0, 6), side="bottom")

        tk.Label(input_frame, text="CMD", bg=PANEL_BG, fg=FG_DIM,
                 font=FONT_SM).pack(side="left", padx=(6, 4))

        self._console_var = tk.StringVar()
        self._console_entry = tk.Entry(
            input_frame, textvariable=self._console_var,
            bg=ACCENT, fg=FG, insertbackground=FG, font=FONT_MONO,
            relief="flat",
        )
        self._console_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._console_entry.bind("<Return>", lambda e: self._on_console_send())

        tk.Button(
            input_frame, text="Send", command=self._on_console_send,
            bg=BTN_HOME, fg=FG, relief="flat", font=FONT_SM,
            width=5, cursor="hand2",
        ).pack(side="left", padx=(0, 6))

        # Output box fills remaining space
        out_frame = tk.Frame(p, bg=PANEL_BG)
        out_frame.pack(fill="both", expand=True, padx=4, pady=(0, 2))

        self._console_out = tk.Text(
            out_frame, bg="#0a0a1a", fg=FG_CONSOLE,
            font=FONT_MONO_SM, relief="flat", state="disabled",
            wrap="word", insertbackground=FG,
        )
        scrollbar = tk.Scrollbar(
            out_frame, command=self._console_out.yview,
            bg=ACCENT, troughcolor=PANEL_BG, width=8,
        )
        self._console_out.configure(yscrollcommand=scrollbar.set)
        self._console_out.pack(side="left", fill="both", expand=True,
                               padx=(4, 0), pady=4)
        scrollbar.pack(side="right", fill="y", pady=4)

    # ── button state management ────────────────────────────────────────────────

    def _set_btn(self, btn, enabled, bg_on, label=None):
        """Enable or disable a button with appropriate colours."""
        if enabled:
            btn.config(state="normal", bg=bg_on, fg=FG,
                       activebackground=bg_on, activeforeground=FG,
                       cursor="hand2")
        else:
            btn.config(state="disabled", bg=BTN_DISABLED, fg=FG_DISABLED,
                       activebackground=BTN_DISABLED, activeforeground=FG_DISABLED,
                       cursor="")
        if label:
            btn.config(text=label)

    def _update_btn_states(self):
        """
        Central place that sets all button states based on current flags.
        Call whenever _homed, _job_running, or _is_paused changes.
        """
        # START / CANCEL
        if not self._homed:
            self._set_btn(self._start_btn, False, BTN_START, "▶  Start")
        elif self._job_running:
            self._set_btn(self._start_btn, True, BTN_CANCEL, "■  Cancel")
        else:
            self._set_btn(self._start_btn, True, BTN_START, "▶  Start")

        # PAUSE / RESUME
        if not self._homed or not self._job_running:
            self._set_btn(self._pause_btn, False, BTN_PAUSE, "⏸  Pause")
        elif self._is_paused:
            self._set_btn(self._pause_btn, True, BTN_PAUSE, "▶  Resume")
        else:
            self._set_btn(self._pause_btn, True, BTN_PAUSE, "⏸  Pause")

        # Jog controls — disabled while job is running (even if paused)
        jog_enabled = not self._job_running
        for b in self._jog_btns:
            if b is self._pen_btn:
                bg = BTN_PEN_DN if self._pen_is_down else BTN_PEN_UP
            else:
                bg = BTN_JOG
            if jog_enabled:
                b.config(state="normal", bg=bg, fg=FG,
                         activebackground=bg, cursor="hand2")
            else:
                b.config(state="disabled", bg=BTN_DISABLED, fg=FG_DISABLED,
                         activebackground=BTN_DISABLED, cursor="")

    # ── port / model helpers ───────────────────────────────────────────────────

    def _refresh_ports(self):
        ports = list_serial_ports()
        self._port_cb["values"] = ports
        if ports:
            self._port_cb.current(0)

    def _refresh_models(self):
        models = scan_models()
        self._model_cb["values"] = models
        if models:
            self._model_cb.current(0)

    # ── connection ─────────────────────────────────────────────────────────────

    def _toggle_connect(self):
        if self.machine.is_connected():
            self.machine.disconnect()
            self._conn_btn.config(text="Connect", bg=BTN_HOME)
            self._homed = False
            self._update_btn_states()
        else:
            port = self._port_var.get()
            if not port:
                messagebox.showerror("No port", "Please select a serial port.")
                return
            try:
                self.machine.connect(port)
                self._conn_btn.config(text="Disconnect", bg=BTN_CANCEL)
            except Exception as e:
                messagebox.showerror("Connect failed", str(e))

    # ── start / cancel ─────────────────────────────────────────────────────────

    def _on_start_cancel(self):
        if self._job_running:
            self._cancel_job()
        else:
            self._start_job()

    def _start_job(self):
        if not self.machine.is_connected():
            messagebox.showwarning("Not connected", "Connect to the machine first.")
            return
        model_name = self._model_var.get()
        if not model_name:
            messagebox.showwarning("No model", "Select a drawing model first.")
            return

        self._stop_model_thread()

        try:
            sys.path.insert(0, MODELS_DIR)
            mod = importlib.import_module(model_name)
            importlib.reload(mod)
        except Exception as e:
            messagebox.showerror("Model error", f"Could not load '{model_name}':\n{e}")
            return

        self._stop_event.clear()
        self.machine.pause_event.set()
        self.machine._abort.clear()       # ensure abort flag is clear for new job
        self._is_paused   = False
        self._job_running = True
        self._update_btn_states()

        self._model_thread = threading.Thread(
            target=mod.run,
            args=(self.machine, self._stop_event, self.machine.pause_event),
            name="ModelThread",
            daemon=True,
        )
        self._model_thread.start()

    def _cancel_job(self):
        if self.machine.is_connected():
            self.machine.stop()           # feed hold + queue STOP_CLEANUP
        self._stop_model_thread()         # wait for model thread to fully die
        # NOTE: do NOT flush queue — STOP_CLEANUP must stay to lift pen and unlock
        self.machine._abort.clear()       # ensure abort flag is clear
        self._job_running = False
        self._is_paused   = False
        self._update_btn_states()

    # ── pause / resume ─────────────────────────────────────────────────────────

    def _on_pause_resume(self):
        if not self.machine.is_connected() or not self._job_running:
            return
        if self._is_paused:
            self.machine.resume()
            self._is_paused = False
        else:
            self.machine.pause()
            self._is_paused = True
        self._update_btn_states()

    # ── model thread ───────────────────────────────────────────────────────────

    def _stop_model_thread(self):
        if self._model_thread and self._model_thread.is_alive():
            self._stop_event.set()
            self.machine.pause_event.set()
            self._model_thread.join(timeout=3)
        self._stop_event.clear()

    # ── homing ─────────────────────────────────────────────────────────────────

    def _on_home(self):
        if not self.machine.is_connected():
            return
        self.machine.home()
        self._homed = True
        self._update_btn_states()

    def _on_set_origin(self):
        if not self.machine.is_connected():
            return
        self.machine.set_origin()

    # ── jog ────────────────────────────────────────────────────────────────────

    def _on_jog(self, dx: int, dy: int):
        if not self.machine.is_connected() or self._job_running:
            return
        try:
            step = float(self._step_var.get())
        except ValueError:
            step = 5.0
        self.machine.jog(dx * step, dy * step)

    # ── pen ────────────────────────────────────────────────────────────────────

    def _on_pen_toggle(self):
        if not self.machine.is_connected() or self._job_running:
            return
        if self._pen_is_down:
            self.machine.pen_up()
            self._pen_is_down = False
        else:
            self.machine.pen_down()
            self._pen_is_down = True
        self._update_btn_states()

    def _on_set_z(self):
        try:
            z = float(self._z_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Z depth must be a number (e.g. -10).")
            return
        self.machine.set_pen_depth(z)

    def _on_push_settings(self):
        if not self.machine.is_connected():
            messagebox.showwarning("Not connected", "Connect to the machine first.")
            return
        self._console_log("--- Pushing settings to firmware ---")
        for key, value in GRBL_SETTINGS.items():
            cmd = f"{key}={value}"
            self._console_log(f"> {cmd}")
            responses = self.machine.send_raw(cmd)
            for line in responses:
                self._console_log(line)
        self._console_log("--- Done ---")

    def _on_download_settings(self):
        if not self.machine.is_connected():
            messagebox.showwarning("Not connected", "Connect to the machine first.")
            return
        self._console_log("--- Reading settings from firmware ---")
        responses = self.machine.send_raw("$$")
        if not responses:
            self._console_log("No response from firmware.")
            return
        # Parse $N=V lines
        parsed = {}
        for line in responses:
            self._console_log(line)
            if line.startswith("$") and "=" in line:
                try:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    # Store as float if possible, else int, else string
                    try:
                        parsed[key] = float(val) if "." in val else int(val)
                    except ValueError:
                        parsed[key] = val
                except Exception:
                    pass
        if not parsed:
            self._console_log("Could not parse any settings.")
            return
        # Read current config.py to preserve software-only settings
        import config as _cfg
        pen_down_z  = self._z_var.get() or str(_cfg.PEN_DOWN_Z)
        jog_step_mm = self._step_var.get() or str(_cfg.JOG_STEP_MM)
        draw_feed   = _cfg.DRAW_FEED
        jog_feed    = _cfg.JOG_FEED
        # Write new config.py
        config_path = os.path.join(os.path.dirname(__file__), "config.py")
        lines = [
            '"""\n',
            'config.py  \u2013  Machine configuration settings.\n',
            '\n',
            'GRBL_SETTINGS are pushed to the firmware via the "Push Settings" button\n',
            'in the config page. They map directly to GRBL $-settings.\n',
            '\n',
            'Other values are used by the Python software only.\n',
            '"""\n',
            '\n',
            '# \u2500\u2500 GRBL firmware settings \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n',
            'GRBL_SETTINGS = {\n',
        ]
        for k, v in sorted(parsed.items(), key=lambda x: int(x[0][1:])):
            lines.append(f'    "{k}": {v},\n')
        lines += [
            '}\n',
            '\n',
            '# \u2500\u2500 Software settings \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n',
            f'PEN_DOWN_Z  = {pen_down_z}\n',
            f'JOG_STEP_MM = {jog_step_mm}\n',
            f'DRAW_FEED   = {draw_feed}\n',
            f'JOG_FEED    = {jog_feed}\n',
        ]
        try:
            with open(config_path, "w") as f:
                f.writelines(lines)
            self._console_log(f"config.py updated with {len(parsed)} settings.")
        except Exception as e:
            self._console_log(f"Failed to write config.py: {e}")

    # ── console ─────────────────────────────────────────────────────────────────

    def _console_log(self, text: str):
        self._console_out.configure(state="normal")
        self._console_out.insert("end", text + "\n")
        self._console_out.see("end")
        self._console_out.configure(state="disabled")

    def _on_console_send(self):
        cmd = self._console_var.get().strip()
        if not cmd or not self.machine.is_connected():
            return
        self._console_log(f"> {cmd}")
        responses = self.machine.send_raw(cmd)
        for line in responses:
            self._console_log(line)
        self._console_var.set("")

    # ── status polling ──────────────────────────────────────────────────────────

    def _poll_status(self):
        s = self.machine.get_status()
        self._status_lbl.config(text=f"Status: {s['state']}")
        if s["state"] != "Disconnected":
            self._x_lbl.config(text=f"{s['x']:+.2f}")
            self._y_lbl.config(text=f"{s['y']:+.2f}")
            if s["pen_down"] != self._pen_is_down and not self._job_running \
                    and not self.machine._in_cleanup:
                self._pen_is_down = s["pen_down"]
                self._update_btn_states()
        else:
            self._x_lbl.config(text="—")
            self._y_lbl.config(text="—")

        # Auto-detect job completion — model thread finished AND machine is idle
        if self._job_running and self._model_thread and \
                not self._model_thread.is_alive():
            grbl_state = s.get("state", "")
            if grbl_state in ("Idle", "Disconnected"):
                self._job_running = False
                self._is_paused   = False
                self._update_btn_states()

        self.after(500, self._poll_status)

    # ── cleanup ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_model_thread()
        if self.machine.is_connected():
            self.machine.disconnect()
        self.destroy()
