"""
models/plants_demo.py  —  Audio-reactive crack-vine drawing model.

Two modes:
  Simulation:  python3 plants_demo.py   (matplotlib visualisation, no machine)
  Drawing:     loaded by the GUI        (sends G-code to the drawing machine)

A web of vines grows outward from the centre of the 190×280mm drawing area,
branching off each other indefinitely.

  BLUE path — smooth sinusoidal wandering driven by ambient audio energy
  RED path  — a sharp straight-line crack triggered by either:
                (a) a fast amplitude transient  (ratio detector)
                (b) a tone re-onset after silence  (re-onset detector)
              The vine continues from the crack tip — pen never lifts mid-vine.

Drawing area: 190 × 280 mm.  Centre = (95, 140).
"""

import math
import random
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

try:
    from scipy.signal import butter, sosfilt, sosfilt_zi
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ===================== DRAWING AREA (mm) =====================
W  = 190.0
H  = 280.0
CX = W / 2    # 95.0
CY = H / 2    # 140.0

PAD_X = W * 0.06   # ~11.4 mm
PAD_Y = H * 0.06   # ~16.8 mm


# ===================== AUDIO SETTINGS =====================
DEVICE_INDEX = None
SR           = 48_000
CHANNELS     = 1
BLOCK        = 512
AUDIO_Q_MAX  = 96


# ===================== DSP =====================
HPF_CUTOFF_HZ         = 35.0
ENABLE_LPF            = True
LPF_CUTOFF_HZ         = 10_000.0
ENABLE_BP_FOR_CONTROL = True
CTRL_BPF_LO_HZ        = 60.0
CTRL_BPF_HI_HZ        = 280.0
CTRL_LPF_HZ           = 10.0
ENV_ALPHA             = 0.06

def db_to_lin(db):
    return 10 ** (db / 20.0)

ENV_GATE = db_to_lin(-30.0)   # vine starts moving above this
ENV_HI   = db_to_lin(-15.0)   # vine hits full speed above this


# ===================== VINE LOCOMOTION =====================
DT        = 1.0 / 60.0

SPEED_MIN = 2.0    # mm/s when quiet
SPEED_MAX = 12.0   # mm/s at loudest

# Wall repulsion — keeps vine inside the safe area
WALL_ZONE  = W * 0.15
WALL_FORCE = 6.0

# Curvature motor
OMEGA_MAX  = 5.2
OMEGA_DAMP = 0.95

# Weave oscillator — gives the vine its sinusoidal organic feel
WEAVE_AMP_MIN = 0.5
WEAVE_AMP_MAX = 3.0
WEAVE_FREQ_HZ = 0.12

# Drift — persistent heading bias, chosen freshly each vine
K_DRIFT    = 1.8

# Audio coupling to curvature
K_AUDIO    = 2.5    # raised to compensate for weak BPF signal in this setup
NOISE_BASE = 0.18
NOISE_GAIN = 1.40

# Min segment length before sending a G-code point to the machine
DRAW_MIN_SEGMENT = 1.0


# ===================== BRANCHING SETTINGS =====================
# A vine ends when EITHER condition is met:
#   1. It has traveled vine_max_dist mm  (randomised per vine)
#   2. VINE_TIMEOUT_S seconds have elapsed (safety net if music is quiet)
# vine_max_dist is drawn from [VINE_MAX_DIST_MIN, VINE_MAX_DIST_MAX].
# The vine must always travel at least VINE_MIN_DIST before either fires.
VINE_MIN_DIST     = 50.0
VINE_MAX_DIST_MIN = 120.0
VINE_MAX_DIST_MAX = 160.0
VINE_TIMEOUT_S    = 20.0

# Branch point sampled from the most recent BRANCH_RECENCY fraction of the vine
BRANCH_RECENCY = 0.5

# Drift candidates constrained 90-180 deg away from previous drift
BRANCH_ANGLE_MIN = math.pi * 0.5
BRANCH_ANGLE_MAX = math.pi * 1.0

# Low-res density grid — biases new drift toward empty page space
DRIFT_GRID_W     = 24
DRIFT_GRID_H     = 35
DRIFT_DEPOSIT    = 1.0
DRIFT_CANDIDATES = 12
DRIFT_LOOKAHEAD  = 80.0   # mm


# ===================== CRACK SETTINGS =====================
CRACK_ENABLE = True

# --- Detector A: fast/slow amplitude transient ---
# Fires on sharp percussive hits or tonal collisions.
CRACK_FAST_TAU  = 0.015   # s — tracks attack quickly
CRACK_SLOW_TAU  = 0.300   # s — tracks background level
CRACK_RATIO_ON  = 1.85    # ratio threshold to trigger
CRACK_RATIO_OFF = 1.30    # ratio must drop below this before re-arming

# --- Detector B: re-onset after silence ---
# Arms when the signal drops into silence, fires when it comes back up.
# This catches clean tone switches that don't produce an amplitude spike.
ONSET_SILENCE_DB  = -30.0   # dBFS — how quiet before arming
ONSET_RETRIG_DB   = -22.0   # dBFS — how loud the comeback must be to fire
ONSET_SILENCE_LIN = db_to_lin(ONSET_SILENCE_DB)
ONSET_RETRIG_LIN  = db_to_lin(ONSET_RETRIG_DB)

# --- Shared crack behaviour ---
# No cracks for this many seconds after a vine starts
CRACK_STARTUP_DELAY_S = 5.0

# Minimum gap between any two cracks (either detector)
CRACK_COOLDOWN_S = 5.0

# Crack geometry
CRACK_LEN_MIN   = W * 0.025       # ~4.75 mm
CRACK_LEN_MAX   = W * 0.060       # ~11.4 mm
CRACK_ANGLE_MIN = math.pi * 0.45  # ~81 deg — near-perpendicular to heading
CRACK_ANGLE_MAX = math.pi * 1.00  # 180 deg — fully backward
CRACK_SPEED     = 12.0            # mm/s — faster than normal vine


# ===================== SMALL HELPERS =====================
def clamp(x, a, b):
    return a if x < a else b if x > b else x

def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def lerp(a, b, t):
    return a + (b - a) * float(clamp(t, 0.0, 1.0))

def alpha_from_tau(tau_s, dt_s):
    return 1.0 - math.exp(-dt_s / max(1e-6, tau_s))

def wall_force(x, y):
    """Gentle quadratic repulsion away from the padded boundary."""
    fx = fy = 0.0
    if x < PAD_X + WALL_ZONE:
        d = (PAD_X + WALL_ZONE) - x
        fx += WALL_FORCE * (d / WALL_ZONE) ** 2
    if x > (W - PAD_X) - WALL_ZONE:
        d = x - ((W - PAD_X) - WALL_ZONE)
        fx -= WALL_FORCE * (d / WALL_ZONE) ** 2
    if y < PAD_Y + WALL_ZONE:
        d = (PAD_Y + WALL_ZONE) - y
        fy += WALL_FORCE * (d / WALL_ZONE) ** 2
    if y > (H - PAD_Y) - WALL_ZONE:
        d = y - ((H - PAD_Y) - WALL_ZONE)
        fy -= WALL_FORCE * (d / WALL_ZONE) ** 2
    return fx, fy


# ===================== FILTERS =====================
class SOSFilter:
    def __init__(self, sos):
        self.sos = np.asarray(sos, dtype=np.float64)
        self.zi  = sosfilt_zi(self.sos) * 0.0
    def process(self, x):
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(np.float32, copy=False)

class OnePoleLP:
    def __init__(self, cutoff_hz, sr):
        self.a  = float(1.0 - np.exp(-2.0 * np.pi * cutoff_hz / sr))
        self.y1 = 0.0
    def process(self, x):
        y  = np.empty_like(x, dtype=np.float32)
        y1 = self.y1; a = self.a
        for i, xn in enumerate(x):
            y1 = y1 + a * (float(xn) - y1)
            y[i] = y1
        self.y1 = y1
        return y

class OnePoleHP:
    def __init__(self, cutoff_hz, sr):
        self.lp = OnePoleLP(cutoff_hz, sr)
    def process(self, x):
        return (x - self.lp.process(x)).astype(np.float32, copy=False)

def build_filters():
    if HAVE_SCIPY:
        hpf      = SOSFilter(butter(2, HPF_CUTOFF_HZ, btype="highpass", fs=SR, output="sos"))
        lpf      = SOSFilter(butter(2, LPF_CUTOFF_HZ, btype="lowpass",  fs=SR, output="sos")) if ENABLE_LPF else None
        bpf      = SOSFilter(butter(2, [CTRL_BPF_LO_HZ, CTRL_BPF_HI_HZ],
                                    btype="bandpass", fs=SR, output="sos")) if ENABLE_BP_FOR_CONTROL else None
        ctrl_lpf = SOSFilter(butter(2, CTRL_LPF_HZ, btype="lowpass", fs=SR, output="sos"))
    else:
        hpf      = OnePoleHP(HPF_CUTOFF_HZ, SR)
        lpf      = OnePoleLP(LPF_CUTOFF_HZ, SR) if ENABLE_LPF else None
        bpf      = None
        ctrl_lpf = OnePoleLP(CTRL_LPF_HZ, SR)
    return hpf, lpf, bpf, ctrl_lpf


# ===================== AUDIO THREAD =====================
audio_q = deque(maxlen=AUDIO_Q_MAX)
q_lock  = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    x = indata[:, 0].astype(np.float32, copy=True)
    with q_lock:
        audio_q.append(x)

def pick_device_index():
    if DEVICE_INDEX is not None:
        return DEVICE_INDEX
    devs       = sd.query_devices()
    best       = None
    default_in = sd.default.device[0] if sd.default.device else None
    for i, d in enumerate(devs):
        if d["max_input_channels"] <= 0:
            continue
        name  = str(d["name"])
        score = sum(10 for tok in ("sabrent", "usb", "audio") if tok in name.lower())
        if default_in is not None and i == default_in:
            score += 2
        if best is None or score > best[0]:
            best = (score, i, name)
    return best[1] if best else None


# ===================== DRIFT DENSITY GRID =====================
_drift_grid = np.zeros((DRIFT_GRID_H, DRIFT_GRID_W), dtype=np.float32)

def _grid_coords(x, y):
    ix = int(np.clip(x / W * DRIFT_GRID_W, 0, DRIFT_GRID_W - 1))
    iy = int(np.clip(y / H * DRIFT_GRID_H, 0, DRIFT_GRID_H - 1))
    return ix, iy

def _deposit_blue_pts(blue_pts):
    for px, py in blue_pts:
        ix, iy = _grid_coords(px, py)
        _drift_grid[iy, ix] += DRIFT_DEPOSIT

def _score_direction(bx, by, angle):
    steps = 8
    total = 0.0
    for i in range(1, steps + 1):
        d  = DRIFT_LOOKAHEAD * i / steps
        sx = float(np.clip(bx + d * math.cos(angle), 0, W))
        sy = float(np.clip(by + d * math.sin(angle), 0, H))
        ix, iy = _grid_coords(sx, sy)
        total += float(_drift_grid[iy, ix])
    return -total


# ===================== BRANCHING HELPERS =====================
def _pick_branch_point(blue_pts):
    n = len(blue_pts)
    if n < 2:
        return None
    start = int(n * (1.0 - BRANCH_RECENCY))
    return blue_pts[random.randint(start, n - 1)]

def _next_drift(bx, by, prev_drift):
    candidates = []
    for i in range(DRIFT_CANDIDATES):
        offset = random.uniform(BRANCH_ANGLE_MIN, BRANCH_ANGLE_MAX)
        sign   = 1 if i % 2 == 0 else -1
        angle  = (prev_drift + sign * offset) % (2 * math.pi)
        candidates.append((_score_direction(bx, by, angle), angle))
    candidates.sort(reverse=True)
    best_angle = candidates[0][1]
    return (best_angle + random.uniform(-0.25, 0.25)) % (2 * math.pi)


# ===================== CORE VINE LOOP =====================
def _vine_loop(on_move, on_stop, on_crack_point, on_branch,
               stop_event, on_crack_start=None):
    """
    Grows an infinite sequence of branching vines.

    Callbacks
    ---------
    on_move(x, y)         -- every point (blue AND red)
    on_stop()             -- loop exiting
    on_crack_point(x, y)  -- crack sprint ended at tip (x, y)
    on_branch(bx, by)     -- vine done; pen moving to branch point (bx, by)
    on_crack_start()      -- optional; fired at crack entry (sim colour tagging)
    stop_event            -- threading.Event; set to halt
    """
    hpf, lpf, bpf, ctrl_lpf = build_filters()

    a_fast  = alpha_from_tau(CRACK_FAST_TAU, DT)
    a_slow  = alpha_from_tau(CRACK_SLOW_TAU, DT)
    prewarm = float(db_to_lin(-30.0))

    device = pick_device_index()
    print(f"[plants] audio device: {device}")

    vine_x      = CX
    vine_y      = CY
    drift_theta = random.uniform(0, 2 * math.pi)
    vine_num    = 0

    with sd.InputStream(device=device, channels=CHANNELS, samplerate=SR,
                        blocksize=BLOCK, dtype="float32",
                        callback=audio_callback, latency="low"):

        while not stop_event.is_set():

            # ----------------------------------------------------------------
            # Initialise a fresh vine
            # ----------------------------------------------------------------
            env_s       = 0.0
            ctrl_s      = 0.0
            env_fast    = prewarm
            env_slow    = prewarm
            crack_armed = True   # transient detector arm state

            # Re-onset detector state
            # Arms when signal drops below ONSET_SILENCE_LIN,
            # fires when signal rises above ONSET_RETRIG_LIN.
            onset_armed = False

            x           = vine_x
            y           = vine_y
            theta       = drift_theta + random.uniform(-0.5, 0.5)
            omega       = random.uniform(-0.3, 0.3)
            eta         = 0.0
            weave_phase = random.uniform(0.0, 2.0 * math.pi)

            state          = "GROW"
            crack_theta    = 0.0
            crack_remain   = 0.0
            crack_cooldown = 0.0

            vine_dist      = 0.0
            vine_max_dist  = random.uniform(VINE_MAX_DIST_MIN, VINE_MAX_DIST_MAX)
            vine_start_t   = time.time()
            last_x, last_y = x, y
            blue_pts       = []

            print(f"[plants] vine {vine_num}  "
                  f"drift={math.degrees(drift_theta):.0f}deg  "
                  f"target={vine_max_dist:.0f}mm")

            # ----------------------------------------------------------------
            # Grow until branch condition
            # ----------------------------------------------------------------
            last_block = np.zeros(BLOCK, dtype=np.float32)

            while not stop_event.is_set():

                # Fetch latest audio block; hold last valid one if queue empty
                with q_lock:
                    if audio_q:
                        last_block = audio_q.pop()
                        audio_q.clear()
                block = last_block

                # ---- DSP ----
                y_audio = hpf.process(block)
                if lpf is not None:
                    y_audio = lpf.process(y_audio)

                env   = float(np.sqrt(np.mean(y_audio ** 2) + 1e-12))
                env_s = (1.0 - ENV_ALPHA) * env_s + ENV_ALPHA * env
                t_env = float(np.clip(
                    (env_s - ENV_GATE) / (ENV_HI - ENV_GATE + 1e-12), 0.0, 1.0))
                base_speed = SPEED_MIN + (SPEED_MAX - SPEED_MIN) * t_env ** 2

                # ---- Detector A: transient (fast/slow ratio) ----
                if CRACK_ENABLE:
                    env_fast = (1.0 - a_fast) * env_fast + a_fast * env
                    env_slow = (1.0 - a_slow) * env_slow + a_slow * env
                    ratio    = env_fast / (env_slow + 1e-9)
                else:
                    ratio = 0.0

                if (not crack_armed) and ratio < CRACK_RATIO_OFF:
                    crack_armed = True

                # ---- Detector B: re-onset after silence ----
                # Step 1: arm when signal drops into silence
                if CRACK_ENABLE and env < ONSET_SILENCE_LIN:
                    onset_armed = True
                # Step 2: if armed and signal comes back up, that's our trigger
                onset_triggered = (CRACK_ENABLE
                                   and onset_armed
                                   and env > ONSET_RETRIG_LIN)
                if onset_triggered:
                    onset_armed = False   # reset until next silence

                # ---- STATE: CRACK ----
                if state == "CRACK":
                    x += CRACK_SPEED * math.cos(crack_theta) * DT
                    y += CRACK_SPEED * math.sin(crack_theta) * DT
                    x  = float(np.clip(x, PAD_X, W - PAD_X))
                    y  = float(np.clip(y, PAD_Y, H - PAD_Y))
                    on_move(x, y)
                    crack_remain -= DT
                    if crack_remain <= 0.0:
                        on_crack_point(x, y)
                        state          = "GROW"
                        crack_cooldown = CRACK_COOLDOWN_S
                        theta          = crack_theta + random.uniform(-0.6, 0.6)
                    time.sleep(DT * 0.5)
                    continue

                # ---- Branch condition ----
                elapsed = time.time() - vine_start_t
                if vine_dist >= vine_max_dist:
                    break
                if vine_dist >= VINE_MIN_DIST and elapsed >= VINE_TIMEOUT_S:
                    break

                # ---- Shared crack trigger gate ----
                startup_ok = elapsed > CRACK_STARTUP_DELAY_S
                if crack_cooldown > 0.0:
                    crack_cooldown -= DT

                fire_crack = False

                if CRACK_ENABLE and startup_ok and crack_cooldown <= 0.0:
                    if crack_armed and ratio > CRACK_RATIO_ON:
                        # Detector A fired
                        crack_armed = False
                        fire_crack  = True
                        print(f"[plants]   crack A (transient)  t={elapsed:.1f}s  ratio={ratio:.2f}")
                    elif onset_triggered:
                        # Detector B fired
                        fire_crack = True
                        print(f"[plants]   crack B (re-onset)   t={elapsed:.1f}s  env={20*math.log10(env+1e-12):.1f}dBFS")

                if fire_crack:
                    state        = "CRACK"
                    angle_offset = random.uniform(CRACK_ANGLE_MIN, CRACK_ANGLE_MAX)
                    crack_theta  = theta + random.choice((-1, 1)) * angle_offset
                    crack_remain = random.uniform(CRACK_LEN_MIN, CRACK_LEN_MAX) / CRACK_SPEED
                    crack_cooldown = CRACK_COOLDOWN_S   # pre-set so cooldown starts now
                    if on_crack_start is not None:
                        on_crack_start()
                    time.sleep(DT * 0.5)
                    continue

                # ---- GROW physics ----
                ctrl = y_audio
                if bpf is not None:
                    ctrl = bpf.process(ctrl)
                ctrl      = ctrl_lpf.process(ctrl)
                ctrl_s    = 0.85 * ctrl_s + 0.15 * float(np.mean(ctrl))
                ctrl_norm = ctrl_s / (env_s + 1e-6)

                drift_align = wrap_pi(drift_theta - theta)

                fx_w, fy_w = wall_force(x, y)
                wall_mag   = math.hypot(fx_w, fy_w)
                wall_align = (wrap_pi(math.atan2(fy_w, fx_w) - theta)
                              if wall_mag > 1e-9 else 0.0)

                weave_phase = (weave_phase + 2.0 * math.pi * WEAVE_FREQ_HZ * DT) % (2.0 * math.pi)
                weave       = lerp(WEAVE_AMP_MIN, WEAVE_AMP_MAX, t_env) * math.sin(weave_phase)

                eta       = 0.985 * eta + 0.015 * random.uniform(-1.0, 1.0)
                noise_amp = NOISE_BASE + NOISE_GAIN * t_env

                omega_target = (K_DRIFT * drift_align
                                + 3.0 * (wall_mag / (WALL_FORCE + 1e-9)) * wall_align
                                + weave
                                + K_AUDIO * ctrl_norm
                                + noise_amp * eta)
                omega = float(np.clip(
                    OMEGA_DAMP * omega + (1.0 - OMEGA_DAMP) * omega_target,
                    -OMEGA_MAX, OMEGA_MAX))

                theta += omega * DT
                x     += base_speed * math.cos(theta) * DT
                y     += base_speed * math.sin(theta) * DT
                x      = float(np.clip(x, PAD_X, W - PAD_X))
                y      = float(np.clip(y, PAD_Y, H - PAD_Y))

                on_move(x, y)

                step = math.hypot(x - last_x, y - last_y)
                vine_dist += step
                last_x, last_y = x, y
                blue_pts.append((x, y))

                time.sleep(DT * 0.5)

            if stop_event.is_set():
                break

            # ----------------------------------------------------------------
            # Branch to next vine
            # ----------------------------------------------------------------
            bp = _pick_branch_point(blue_pts) or (x, y)
            on_branch(bp[0], bp[1])
            _deposit_blue_pts(blue_pts)

            vine_x      = bp[0]
            vine_y      = bp[1]
            drift_theta = _next_drift(bp[0], bp[1], drift_theta)
            vine_num   += 1

    on_stop()


# ===================== SIMULATION MODE =====================
def run_simulation():
    """Standalone matplotlib visualisation — no machine needed."""
    import queue as _queue

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    TRAIL_LEN  = int(300 / DT)
    stop_event = threading.Event()
    pt_queue   = _queue.Queue(maxsize=4096)
    in_crack   = [False]

    def on_move(x, y):
        try:
            pt_queue.put_nowait(('R' if in_crack[0] else 'B', x, y))
        except _queue.Full:
            pass

    def on_crack_start():
        in_crack[0] = True

    def on_crack_point(x, y):
        in_crack[0] = False

    def on_branch(bx, by):
        in_crack[0] = False
        try:
            pt_queue.put_nowait(('N', float('nan'), float('nan')))
        except _queue.Full:
            pass

    def on_stop():
        pass

    t = threading.Thread(
        target=_vine_loop,
        args=(on_move, on_stop, on_crack_point, on_branch, stop_event),
        kwargs={"on_crack_start": on_crack_start},
        daemon=True,
    )
    t.start()

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_title("Plants Demo – Simulation")
    ax.set_xlim(-5, W + 5)
    ax.set_ylim(-5, H + 5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    ax.add_patch(mpatches.Rectangle(
        (0, 0), W, H,
        linewidth=1.5, edgecolor="#555555", facecolor="none", linestyle="--"))
    ax.add_patch(mpatches.Rectangle(
        (PAD_X, PAD_Y), W - 2*PAD_X, H - 2*PAD_Y,
        linewidth=0.8, edgecolor="#aaaaaa", facecolor="none", linestyle=":"))

    (blue_line,) = ax.plot([], [], lw=1.5, alpha=0.9, color="#2255cc")
    (red_line,)  = ax.plot([], [], lw=2.0, alpha=0.9, color="#cc2222")

    trail_tags: deque = deque(maxlen=TRAIL_LEN)
    trail_x:    deque = deque(maxlen=TRAIL_LEN)
    trail_y:    deque = deque(maxlen=TRAIL_LEN)
    last_draw = time.time()

    try:
        while t.is_alive():
            while not pt_queue.empty():
                try:
                    tag, px, py = pt_queue.get_nowait()
                    trail_tags.append(tag)
                    trail_x.append(px)
                    trail_y.append(py)
                except _queue.Empty:
                    break

            if time.time() - last_draw > 1.0 / 30.0:
                tags = list(trail_tags)
                txs  = list(trail_x)
                tys  = list(trail_y)

                bx: list = []; by: list = []
                rx: list = []; ry: list = []
                prev = None

                for tag, px, py in zip(tags, txs, tys):
                    if tag == 'N':
                        bx.append(float('nan')); by.append(float('nan'))
                        rx.append(float('nan')); ry.append(float('nan'))
                        prev = None
                        continue
                    if tag == 'B':
                        if prev == 'R':
                            bx.append(float('nan')); by.append(float('nan'))
                        bx.append(px); by.append(py)
                    else:
                        if prev == 'B':
                            rx.append(float('nan')); ry.append(float('nan'))
                        rx.append(px); ry.append(py)
                    prev = tag

                blue_line.set_data(bx, by)
                red_line.set_data(rx, ry)
                fig.canvas.draw()
                fig.canvas.flush_events()
                last_draw = time.time()

            time.sleep(0.01)

    except KeyboardInterrupt:
        stop_event.set()
        t.join(timeout=3)

    plt.ioff()
    plt.show()


# ===================== DRAWING MODE =====================
def run(machine, stop_event: threading.Event, pause_event: threading.Event) -> None:
    """
    Called by the GUI.
    Within a vine: pen never lifts (blue and red share one continuous stroke).
    Between vines: pen lifts, G0 rapid to branch point, then pen down on the
    first qualifying move of the new vine (once it has traveled DRAW_MIN_SEGMENT
    from the branch point).
    """
    from machine import draw_to_machine, DRAW_FEED

    last_pos = [None]

    def _send(x, y):
        lp = last_pos[0]
        mx, my = draw_to_machine(x, y)
        if lp is None:
            machine.send_gcode(f"G0 X{mx:.3f} Y{my:.3f}")
            machine.pen_down()
            last_pos[0] = (x, y)
            return
        if math.hypot(x - lp[0], y - lp[1]) >= DRAW_MIN_SEGMENT:
            machine.send_gcode(f"G1 X{mx:.3f} Y{my:.3f} F{DRAW_FEED}")
            last_pos[0] = (x, y)

    def on_move(x, y):
        if not pause_event.is_set():
            machine.pen_up()
            pause_event.wait()
            if stop_event.is_set():
                return
            time.sleep(0.1)
        if stop_event.is_set():
            return
        _send(x, y)

    def on_crack_point(x, y):
        pass  # tip already drawn via on_move

    def on_branch(bx, by):
        """Lift pen, rapid to branch point — pen down on first qualifying move."""
        if stop_event.is_set():
            return
        machine.pen_up()
        mx, my = draw_to_machine(bx, by)
        machine.send_gcode(f"G0 X{mx:.3f} Y{my:.3f}")
        last_pos[0] = None

    def on_stop():
        machine.pen_up()

    _vine_loop(on_move, on_stop, on_crack_point, on_branch, stop_event)


# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    run_simulation()
