"""
config.py  –  Machine configuration settings.

GRBL_SETTINGS are pushed to the firmware via the "Push Settings" button
in the config page. They map directly to GRBL $-settings.

Other values are used by the Python software only.
"""

# ── GRBL firmware settings ─────────────────────────────────────────────────────
# These are sent to GRBL via $N=V commands when you press "Push Settings"
GRBL_SETTINGS = {
    # Steps per mm
    "$100": 50.0,    # X steps/mm
    "$101": 50.0,    # Y steps/mm
    "$102": 50.0,    # Z steps/mm

    # Max feed rates (mm/min)
    "$110": 2500.0,  # X max rate
    "$111": 2500.0,  # Y max rate
    "$112": 2500.0,  # Z max rate

    # Acceleration (mm/s^2)
    "$120": 150.0,   # X acceleration
    "$121": 150.0,   # Y acceleration
    "$122": 150.0,   # Z acceleration

    # Travel limits (mm)
    "$130": 210.0,   # X max travel
    "$131": 300.0,   # Y max travel
    "$132": 10.0,    # Z max travel

    # Homing
    "$22": 1,        # Homing enabled
    "$23": 3,        # Homing direction invert mask
    "$24": 200.0,    # Homing locate feed rate
    "$25": 600.0,    # Homing seek feed rate
    "$26": 10,       # Homing debounce (ms)
    "$27": 3.0,      # Homing pull-off (mm)

    # Soft limits & hard limits
    "$20": 1,        # Soft limits enabled
    "$21": 0,        # Hard limits disabled

    # Direction invert
    "$3":  0,        # Step direction invert mask

    # Other
    "$1":  255,      # Step idle delay (ms) — keep motors energised
    "$11": 0.010,    # Junction deviation
    "$12": 0.002,    # Arc tolerance
    "$13": 0,        # Report in mm
}

# ── Software settings ──────────────────────────────────────────────────────────
PEN_DOWN_Z    = -10.0   # Default Z depth for pen down (mm)
JOG_STEP_MM   = 10.0    # Default jog step size (mm)
DRAW_FEED     = 800     # Drawing feed rate (mm/min)
JOG_FEED      = 1000    # Jog feed rate (mm/min)
