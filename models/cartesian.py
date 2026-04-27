"""
models/cartesian.py  –  Draws a cartesian origin marker at the centre of the
drawing area:
  - A horizontal line 60mm wide (±30mm from centre)
  - A vertical line 60mm tall (±30mm from centre)
  - A small filled circle (concentric rings) at centre

Each element is treated as one logical drawing unit — pause is only checked
between units, never inside a point loop. Follows the same pattern as square.py.

Drawing area: 190 x 280 mm, origin = top-left of safe area.
Centre = (95, 140).
"""

import math
import time
import threading

from machine import draw_to_machine, DRAW_FEED, DRAW_W, DRAW_H

# ── settings ──────────────────────────────────────────────────────────────────
CX        = DRAW_W / 2   # 95.0
CY        = DRAW_H / 2   # 140.0
LINE_HALF = 30.0          # half-length of each axis line (total 60mm)
CIRCLE_R  = 3.0           # outer radius of filled circle (mm)
RING_STEP = 0.4           # spacing between concentric fill rings (mm)
SEGMENTS  = 32            # polygon segments per ring


def _circle_points(cx, cy, r):
    pts = []
    for i in range(SEGMENTS + 1):
        angle = math.radians(i / SEGMENTS * 360.0)
        pts.append((cx + r * math.cos(angle),
                    cy + r * math.sin(angle)))
    return pts


def run(machine, stop_event: threading.Event, pause_event: threading.Event) -> None:

    def check_pause():
        """
        Call between logical drawing units only — never inside a point loop.
        Lifts pen and blocks until resumed. Returns False if stopped.
        """
        if not pause_event.is_set():
            machine.pen_up()
            pause_event.wait()
            if stop_event.is_set():
                return False
            time.sleep(0.1)   # let GRBL finish cycle-resume
        return not stop_event.is_set()

    def move_to(x, y):
        mx, my = draw_to_machine(x, y)
        machine.send_gcode(f"G0 X{mx:.3f} Y{my:.3f}")

    def draw_to(x, y):
        mx, my = draw_to_machine(x, y)
        machine.send_gcode(f"G1 X{mx:.3f} Y{my:.3f} F{DRAW_FEED}")

    # ── 1. Horizontal line ─────────────────────────────────────────────────────
    if not check_pause(): return
    move_to(CX - LINE_HALF, CY)
    machine.pen_down()
    draw_to(CX + LINE_HALF, CY)
    machine.pen_up()

    # ── 2. Vertical line ───────────────────────────────────────────────────────
    if not check_pause(): return
    move_to(CX, CY - LINE_HALF)
    machine.pen_down()
    draw_to(CX, CY + LINE_HALF)
    machine.pen_up()

    # ── 3. Filled circle (one ring = one drawing unit) ────────────────────────
    r = CIRCLE_R
    while r > 0 and not stop_event.is_set():
        if not check_pause(): return

        pts = _circle_points(CX, CY, r)
        move_to(pts[0][0], pts[0][1])
        machine.pen_down()
        for (x, y) in pts[1:]:
            draw_to(x, y)
        machine.pen_up()

        r = round(r - RING_STEP, 6)

    # ── done ──────────────────────────────────────────────────────────────────
    machine.pen_up()
    move_to(CX, CY)

    # Wait for GRBL to finish executing all queued commands before exiting.
    # Prevents the GUI from marking the job done while the machine is still
    # drawing — same approach as the PAUSE_BETWEEN wait in square.py.
    deadline = time.time() + 60.0
    while time.time() < deadline:
        if stop_event.is_set():
            break
        s = machine.get_status()
        if s.get("state") == "Idle" and machine._cmd_queue.empty():
            break
        time.sleep(0.2)
