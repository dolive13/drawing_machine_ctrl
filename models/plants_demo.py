"""
models/cracks.py  –  Audio-reactive crack-vine drawing model.

Two modes:
  Simulation:  python3 cracks.py          (matplotlib visualisation, no machine)
  Drawing:     loaded by GUI as a model   (sends G-code to machine)

A single vine grows outward from the centre of the 190×280mm drawing area.
  BLUE path  — smooth sinusoidal wandering driven by ambient audio energy
  RED path   — a sharp straight-line crack triggered by transient hits;
               the vine continues growing from the crack tip (pen never lifts)

Drawing area: 190 x 280 mm, origin = top-left of safe area.
Centre = (95, 140).
"""

import time
import math
import random
import threading
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

ENV_GATE = db_to_lin(-45.0)
ENV_HI   = db_to_lin(-25.0)


# ===================== VINE LOCOMOTION =====================
DT        = 1.0 / 60.0

SPEED_MIN = 2.0    # mm/s when quiet
SPEED_MAX = 12.0   # mm/s at loudest

# Wall repulsion — keeps vine inside the safe area
WALL_ZONE  = W * 0.15   # repulsion starts this far from the padded edge
WALL_FORCE = 6.0

# Curvature motor
OMEGA_MAX  = 5.2
OMEGA_DAMP = 0.95

# Weave oscillator — gives the vine its sinusoidal organic feel
WEAVE_AMP_MIN = 0.5
WEAVE_AMP_MAX = 3.0
WEAVE_FREQ_HZ = 0.12

# Drift — a persistent heading bias chosen randomly at startup.
# K_DRIFT controls how strongly the vine is pulled back toward its
# drift direction.  Higher = straighter; lower = more wandering.
K_DRIFT    = 1.8

# Audio coupling to curvature
K_AUDIO    = 1.35
NOISE_BASE = 0.18
NOISE_GAIN = 1.40

# Drawing: min segment length before sending a G-code point
DRAW_MIN_SEGMENT = 1.0


# ===================== BRANCHING SETTINGS =====================
# A vine ends when EITHER condition is met:
#   1. It has traveled at least vine_max_dist mm (randomised per vine)
#   2. A timeout of VINE_TIMEOUT_S seconds has elapsed (safety net)
# vine_max_dist is randomised in [VINE_MAX_DIST_MIN, VINE_MAX_DIST_MAX]
# but the vine must always travel at least VINE_MIN_DIST before it can end.
VINE_MIN_DIST      = 50.0    # mm — minimum before any end condition applies
VINE_MAX_DIST_MIN  = 120.0   # mm — shortest randomised target length
VINE_MAX_DIST_MAX  = 160.0   # mm — longest randomised target length
VINE_TIMEOUT_S     = 20.0    # s  — hard timeout if max dist never reached

# Branch point is sampled from the recent portion of the previous vine's
# blue-point history.  0.5 = second half only, 1.0 = tip only.
BRANCH_RECENCY = 0.5

# New drift direction is nudged away from the previous drift by an angle
# in this range (radians), left or right randomly.
BRANCH_ANGLE_MIN = math.pi * 0.5   # ~90 deg minimum spread
BRANCH_ANGLE_MAX = math.pi * 1.0   # 180 deg maximum spread

# Drift direction heatmap — sampled once per vine at branch time.
# Low-res grid tracks cumulative path density across the page.
# Candidate drift angles are scored by how much unvisited space lies
# ahead, and the best one wins (with a small random nudge for variety).
DRIFT_GRID_W    = 24          # grid columns
DRIFT_GRID_H    = 35          # grid rows (roughly square cells on 190x280)
DRIFT_DEPOSIT   = 1.0         # density added per blue point
DRIFT_CANDIDATES = 12         # number of angles sampled per branch decision
DRIFT_LOOKAHEAD  = 80.0       # mm ahead to score each candidate direction


# ===================== CRACK SETTINGS =====================
CRACK_ENABLE = True

# Transient detector: fast/slow envelope ratio
CRACK_FAST_TAU  = 0.015   # s  — tracks attack quickly
CRACK_SLOW_TAU  = 0.300   # s  — tracks background level
CRACK_RATIO_ON  = 1.85    # ratio threshold to trigger a crack
CRACK_RATIO_OFF = 1.30    # ratio must fall below this before re-arming

# No cracks for this many seconds after the vine starts
CRACK_STARTUP_DELAY_S = 5.0

# Minimum gap between cracks (seconds) — prevents red paths dominating
# in percussive music.  Tune this to taste: 4–8s works well for most music.
CRACK_COOLDOWN_S = 5.0   # long enough for env_slow to fully warm up

# Crack length range (mm) — chosen randomly each trigger
CRACK_LEN_MIN = W * 0.025   # ~4.75 mm
CRACK_LEN_MAX = W * 0.060   # ~11.4 mm

# Crack direction bias: angle offset from vine heading (radians).
# 0 = forward, pi = fully backward.  Range kept away from 0 so cracks
# always go sideways-to-backward, never forward.
CRACK_ANGLE_MIN = math.pi * 0.45   # ~81 deg — near-perpendicular
CRACK_ANGLE_MAX = math.pi * 1.00   # 180 deg — fully backward
# Left/right side is chosen randomly each crack.

# Speed during the crack sprint (mm/s) — noticeably faster than normal vine
CRACK_SPEED = 12.0


# ===================== HELPERS =====================
def clamp(x, a, b):
    return a if x < a else b if x > b else x

def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def lerp(a, b, t):
    return a + (b - a) * float(clamp(t, 0.0, 1.0))

def alpha_from_tau(tau_s, dt_s):
    return 1.0 - math.exp(-dt_s / max(1e-6, tau_s))

def wall_force(x, y):
    """Gentle repulsion away from the padded boundary."""
    fx = fy = 0.0
    if x < PAD_X + WALL_ZONE:
        d   = (PAD_X + WALL_ZONE) - x
        fx += WALL_FORCE * (d / WALL_ZONE) ** 2
    if x > (W - PAD_X) - WALL_ZONE:
        d   = x - ((W - PAD_X) - WALL_ZONE)
        fx -= WALL_FORCE * (d / WALL_ZONE) ** 2
    if y < PAD_Y + WALL_ZONE:
        d   = (PAD_Y + WALL_ZONE) - y
        fy += WALL_FORCE * (d / WALL_ZONE) ** 2
    if y > (H - PAD_Y) - WALL_ZONE:
        d   = y - ((H - PAD_Y) - WALL_ZONE)
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
            y1   = y1 + a * (float(xn) - y1)
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
        return hpf, lpf, bpf, ctrl_lpf
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
# Shared across all vines — accumulates where the drawing has been.
# Updated after each vine finishes; read at branch time to pick drift.
_drift_grid = np.zeros((DRIFT_GRID_H, DRIFT_GRID_W), dtype=np.float32)

def _grid_coords(x, y):
    """Convert mm position to grid cell indices."""
    ix = int(np.clip(x / W * DRIFT_GRID_W, 0, DRIFT_GRID_W - 1))
    iy = int(np.clip(y / H * DRIFT_GRID_H, 0, DRIFT_GRID_H - 1))
    return ix, iy

def _deposit_blue_pts(blue_pts):
    """Stamp all blue points from a finished vine into the density grid."""
    for (px, py) in blue_pts:
        ix, iy = _grid_coords(px, py)
        _drift_grid[iy, ix] += DRIFT_DEPOSIT

def _score_direction(bx, by, angle):
    """
    Score a candidate drift angle by sampling density along a ray
    from (bx, by).  Lower density ahead = higher (better) score.
    """
    steps  = 8
    total  = 0.0
    for i in range(1, steps + 1):
        d  = DRIFT_LOOKAHEAD * i / steps
        sx = bx + d * math.cos(angle)
        sy = by + d * math.sin(angle)
        sx = float(np.clip(sx, 0, W))
        sy = float(np.clip(sy, 0, H))
        ix, iy = _grid_coords(sx, sy)
        total += float(_drift_grid[iy, ix])
    return -total   # lower density = higher score


# ===================== BRANCHING HELPERS =====================
def _pick_branch_point(blue_pts):
    """
    Pick a branch point from the recent portion of a vine's blue-point list.
    Returns (x, y) or None if the list is too short.
    """
    n = len(blue_pts)
    if n < 2:
        return None
    start_idx = int(n * (1.0 - BRANCH_RECENCY))
    idx = random.randint(start_idx, n - 1)
    return blue_pts[idx]

def _next_drift(bx, by, prev_drift):
    """
    Choose the next drift direction by:
      1. Sampling DRIFT_CANDIDATES angles, each nudged 90–180° away from
         prev_drift (alternating left/right so we cover both sides evenly).
      2. Scoring each by how much unvisited space lies ahead in the grid.
      3. Picking the best, plus a small random jitter for variety.
    """
    candidates = []
    for i in range(DRIFT_CANDIDATES):
        offset    = random.uniform(BRANCH_ANGLE_MIN, BRANCH_ANGLE_MAX)
        sign      = 1 if i % 2 == 0 else -1
        angle     = (prev_drift + sign * offset) % (2 * math.pi)
        score     = _score_direction(bx, by, angle)
        candidates.append((score, angle))

    candidates.sort(key=lambda c: c[0], reverse=True)
    best_score, best_angle = candidates[0]

    # Small jitter so identical density areas don't always pick the same angle
    jitter = random.uniform(-0.25, 0.25)
    return (best_angle + jitter) % (2 * math.pi)


# ===================== CORE VINE LOOP =====================
def _vine_loop(on_move, on_stop, on_crack_point, on_branch, stop_event):
    """
    Shared vine loop used by both simulation and drawing modes.

    on_move(x, y)         — vine moved to (x, y) in mm; called for ALL points
    on_stop()             — loop is exiting
    on_crack_point(x, y)  — crack sprint just ended at (x, y)
    on_branch(bx, by)     — pen is about to lift and move to branch point (bx, by);
                            drawing mode uses this to send the rapid G0 move
    stop_event            — threading.Event; set to halt
    """
    hpf, lpf, bpf, ctrl_lpf = build_filters()

    a_fast   = alpha_from_tau(CRACK_FAST_TAU, DT)
    a_slow   = alpha_from_tau(CRACK_SLOW_TAU, DT)
    _prewarm = float(db_to_lin(-30.0))

    device = pick_device_index()
    print(f"[cracks] audio device: {device}")

    # Vine 0 starts at page centre with a fully random drift
    vine_x      = CX
    vine_y      = CY
    drift_theta = random.uniform(0, 2 * math.pi)
    vine_num    = 0

    with sd.InputStream(device=device, channels=CHANNELS, samplerate=SR,
                        blocksize=BLOCK, dtype="float32",
                        callback=audio_callback, latency="low"):

        while not stop_event.is_set():
            # ----------------------------------------------------------------
            # Set up a fresh vine
            # ----------------------------------------------------------------
            env_s       = 0.0
            ctrl_s      = 0.0
            env_fast    = _prewarm
            env_slow    = _prewarm
            crack_armed = True

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

            # Vine lifetime
            vine_dist_traveled = 0.0
            vine_max_dist      = random.uniform(VINE_MAX_DIST_MIN, VINE_MAX_DIST_MAX)
            vine_start_t       = time.time()
            last_x, last_y     = x, y

            # Accumulate blue points for branch-point picking
            blue_pts = []

            print(f"[cracks] vine {vine_num}  drift={math.degrees(drift_theta):.0f}°  "
                  f"target={vine_max_dist:.0f}mm")

            # ----------------------------------------------------------------
            # Grow this vine until branch conditions are met
            # ----------------------------------------------------------------
            last_block = np.zeros(BLOCK, dtype=np.float32)   # held across ticks
            while not stop_event.is_set():

                with q_lock:
                    if audio_q:
                        last_block = audio_q.pop()
                        audio_q.clear()
                block = last_block

                y_audio = hpf.process(block)
                if lpf is not None:
                    y_audio = lpf.process(y_audio)

                env   = float(np.sqrt(np.mean(y_audio.astype(np.float32) ** 2) + 1e-12))
                env_s = (1.0 - ENV_ALPHA) * env_s + ENV_ALPHA * env
                t_env = float(np.clip((env_s - ENV_GATE) / (ENV_HI - ENV_GATE + 1e-12), 0.0, 1.0))

                base_speed = SPEED_MIN + (SPEED_MAX - SPEED_MIN) * (t_env * t_env)

                if CRACK_ENABLE:
                    env_fast = (1.0 - a_fast) * env_fast + a_fast * env
                    env_slow = (1.0 - a_slow) * env_slow + a_slow * env
                    ratio    = env_fast / (env_slow + 1e-9)
                else:
                    ratio = 0.0

                if (not crack_armed) and (ratio < CRACK_RATIO_OFF):
                    crack_armed = True

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

                # ---- Check branch condition ----
                elapsed = time.time() - vine_start_t
                if vine_dist_traveled >= vine_max_dist:
                    break   # hit target distance
                if vine_dist_traveled >= VINE_MIN_DIST and elapsed >= VINE_TIMEOUT_S:
                    break   # timeout safety net

                # ---- Crack trigger ----
                startup_ok = elapsed > CRACK_STARTUP_DELAY_S
                if crack_cooldown > 0.0:
                    crack_cooldown -= DT

                if CRACK_ENABLE and crack_armed and startup_ok and crack_cooldown <= 0.0 and ratio > CRACK_RATIO_ON:
                    crack_armed  = False
                    state        = "CRACK"
                    angle_offset = random.uniform(CRACK_ANGLE_MIN, CRACK_ANGLE_MAX)
                    side_sign    = random.choice((-1, 1))
                    crack_theta  = theta + side_sign * angle_offset
                    crack_len    = random.uniform(CRACK_LEN_MIN, CRACK_LEN_MAX)
                    crack_remain = crack_len / CRACK_SPEED
                    time.sleep(DT * 0.5)
                    continue

                # ---- GROW physics ----
                ctrl = y_audio
                if bpf is not None:
                    ctrl = bpf.process(ctrl)
                ctrl      = ctrl_lpf.process(ctrl)
                c         = float(np.mean(ctrl))
                ctrl_s    = 0.85 * ctrl_s + 0.15 * c
                ctrl_norm = ctrl_s / (env_s + 1e-6)

                drift_align = wrap_pi(drift_theta - theta)

                fx_w, fy_w = wall_force(x, y)
                wall_mag   = math.hypot(fx_w, fy_w)
                wall_align = wrap_pi(math.atan2(fy_w, fx_w) - theta) if wall_mag > 1e-9 else 0.0

                weave_phase = (weave_phase + 2.0 * math.pi * WEAVE_FREQ_HZ * DT) % (2.0 * math.pi)
                weave_amp   = lerp(WEAVE_AMP_MIN, WEAVE_AMP_MAX, t_env)
                weave       = weave_amp * math.sin(weave_phase)

                eta       = 0.985 * eta + 0.015 * random.uniform(-1.0, 1.0)
                noise_amp = NOISE_BASE + NOISE_GAIN * t_env

                omega_target = (K_DRIFT * drift_align
                                + 3.0 * (wall_mag / (WALL_FORCE + 1e-9)) * wall_align
                                + weave
                                + K_AUDIO * ctrl_norm
                                + noise_amp * eta)
                omega = float(np.clip(OMEGA_DAMP * omega + (1.0 - OMEGA_DAMP) * omega_target,
                                      -OMEGA_MAX, OMEGA_MAX))

                theta += omega * DT
                x     += base_speed * math.cos(theta) * DT
                y     += base_speed * math.sin(theta) * DT
                x      = float(np.clip(x, PAD_X, W - PAD_X))
                y      = float(np.clip(y, PAD_Y, H - PAD_Y))

                on_move(x, y)

                # Track distance and store blue point
                step = math.hypot(x - last_x, y - last_y)
                vine_dist_traveled += step
                last_x, last_y = x, y
                blue_pts.append((x, y))

                time.sleep(DT * 0.5)

            if stop_event.is_set():
                break

            # ----------------------------------------------------------------
            # Branch: pick a point on the previous vine, move there, start new
            # ----------------------------------------------------------------
            bp = _pick_branch_point(blue_pts)
            if bp is None:
                bp = (x, y)   # fallback: branch from tip

            bx, by = bp
            on_branch(bx, by)

            # Stamp this vine's path into the density grid, then pick
            # a drift direction that points toward emptier space
            _deposit_blue_pts(blue_pts)

            vine_x      = bx
            vine_y      = by
            drift_theta = _next_drift(bx, by, drift_theta)
            vine_num   += 1

    on_stop()


# ===================== SIMULATION MODE =====================
def run_simulation():
    """Standalone matplotlib simulation — no machine needed."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import queue as _queue

    TRAIL_LEN = int(300 / DT)   # ~5 min of history on screen

    stop_event = threading.Event()
    pt_queue   = _queue.Queue(maxsize=2048)

    # in_crack[0] is set True by the loop when a crack starts,
    # and False when on_crack_point fires (crack ends).
    # Because _vine_loop now sets in_crack via on_crack_point and a
    # separate flag we inject it through the callbacks below.
    in_crack = [False]

    def on_move(x, y):
        tag = 'R' if in_crack[0] else 'B'
        try:
            pt_queue.put_nowait((tag, x, y))
        except _queue.Full:
            pass

    def on_crack_point(x, y):
        in_crack[0] = False

    def on_branch(bx, by):
        # Insert a NaN break so matplotlib draws a gap at the branch jump
        try:
            pt_queue.put_nowait(('N', float('nan'), float('nan')))
        except _queue.Full:
            pass

    def on_stop():
        pass

    # _vine_loop now sets in_crack[0]=True when it enters CRACK state.
    # We need that hook — wrap on_move to let the loop signal crack start
    # by passing in_crack as a closure the loop can mutate directly.
    # Cleanest: subclass via a thin wrapper that passes in_crack through.
    def _loop_wrapper():
        # Shadow the module-level _vine_loop with in_crack injection.
        # We do this by patching on_move to check a shared flag that the
        # loop sets just before entering CRACK state.
        # The loop already calls on_crack_point at CRACK end — we just need
        # the start signal.  We inject it by passing a crack_flag list into
        # a monkey-patched version of _vine_loop inline.

        import types

        # Re-use _vine_loop but intercept the CRACK state entry to set in_crack.
        # Simplest: override on_move with a wrapper that checks a shared state
        # flag written by a patched on_crack callback chain — but that's circular.
        #
        # Cleanest solution: pass in_crack directly to _vine_loop so it can
        # set it.  We do this by adding an extra kwarg the loop accepts.
        # Since _vine_loop doesn't support that yet, we replicate the crack-start
        # tagging here by wrapping on_crack_point to also clear in_crack and
        # providing a crack_start hook via a modified call.
        #
        # Actually the simplest correct solution: just call _vine_loop with a
        # patched on_move that reads from a shared flag, and pass a
        # set_in_crack callback that the loop calls on crack entry.
        # _vine_loop already supports this via on_crack_point for the END.
        # We add crack_start support by passing a mutable dict.

        flag = in_crack   # shared mutable list [bool]

        def _on_move_tagged(x, y):
            tag = 'R' if flag[0] else 'B'
            try:
                pt_queue.put_nowait((tag, x, y))
            except _queue.Full:
                pass

        def _on_crack_point(x, y):
            flag[0] = False

        def _on_branch(bx, by):
            flag[0] = False   # ensure clean state after branch
            try:
                pt_queue.put_nowait(('N', float('nan'), float('nan')))
            except _queue.Full:
                pass

        def _on_stop():
            pass

        # _vine_loop sets in_crack True when CRACK state starts but we have
        # no hook for that yet in the shared loop.  The loop does so via the
        # crack trigger block — we surface it by passing in_crack as a side
        # channel through a closure.  Since _vine_loop is defined in this
        # module we can just access in_crack directly from the loop if we
        # restructure — but to avoid another rewrite we use a simpler trick:
        # wrap on_crack_point to set flag False (done above), and detect
        # crack START by checking that on_move is called after a
        # non-B-tagged gap.  The cleanest way: just add a fifth callback.
        # We do that now by calling a local version of _vine_loop that
        # accepts on_crack_start.

        _vine_loop_with_start(_on_move_tagged, _on_stop, _on_crack_point,
                              _on_branch, flag, stop_event)

    t = threading.Thread(target=_loop_wrapper, daemon=True)
    t.start()

    # ---- matplotlib setup ----
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 9))
    ax.set_title("Crack Vine – Simulation")
    ax.set_xlim(-5, W + 5)
    ax.set_ylim(-5, H + 5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    boundary = mpatches.Rectangle((0, 0), W, H,
                                   linewidth=1.5, edgecolor="#555555",
                                   facecolor="none", linestyle="--")
    ax.add_patch(boundary)
    safe = mpatches.Rectangle((PAD_X, PAD_Y), W - 2*PAD_X, H - 2*PAD_Y,
                               linewidth=0.8, edgecolor="#aaaaaa",
                               facecolor="none", linestyle=":")
    ax.add_patch(safe)

    (blue_line,) = ax.plot([], [], linewidth=1.5, alpha=0.9, color="#2255cc")
    (red_line,)  = ax.plot([], [], linewidth=2.0, alpha=0.9, color="#cc2222")

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
                    if tag == 'N':   # branch gap
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


def _vine_loop_with_start(on_move, on_stop, on_crack_point, on_branch,
                          in_crack_flag, stop_event):
    """
    Identical to _vine_loop but also sets in_crack_flag[0] = True
    at crack entry (for simulation colour tagging) and accepts on_branch.
    """
    hpf, lpf, bpf, ctrl_lpf = build_filters()

    a_fast   = alpha_from_tau(CRACK_FAST_TAU, DT)
    a_slow   = alpha_from_tau(CRACK_SLOW_TAU, DT)
    _prewarm = float(db_to_lin(-30.0))

    device = pick_device_index()
    print(f"[cracks-sim] audio device: {device}")

    vine_x      = CX
    vine_y      = CY
    drift_theta = random.uniform(0, 2 * math.pi)
    vine_num    = 0

    with sd.InputStream(device=device, channels=CHANNELS, samplerate=SR,
                        blocksize=BLOCK, dtype="float32",
                        callback=audio_callback, latency="low"):

        while not stop_event.is_set():

            env_s       = 0.0
            ctrl_s      = 0.0
            env_fast    = _prewarm
            env_slow    = _prewarm
            crack_armed = True

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

            vine_dist_traveled = 0.0
            vine_max_dist      = random.uniform(VINE_MAX_DIST_MIN, VINE_MAX_DIST_MAX)
            vine_start_t       = time.time()
            last_x, last_y     = x, y
            blue_pts           = []

            print(f"[cracks-sim] vine {vine_num}  drift={math.degrees(drift_theta):.0f}°  "
                  f"target={vine_max_dist:.0f}mm")

            last_block = np.zeros(BLOCK, dtype=np.float32)
            while not stop_event.is_set():

                with q_lock:
                    if audio_q:
                        last_block = audio_q.pop()
                        audio_q.clear()
                block = last_block

                y_audio = hpf.process(block)
                if lpf is not None:
                    y_audio = lpf.process(y_audio)

                env   = float(np.sqrt(np.mean(y_audio.astype(np.float32) ** 2) + 1e-12))
                env_s = (1.0 - ENV_ALPHA) * env_s + ENV_ALPHA * env
                t_env = float(np.clip((env_s - ENV_GATE) / (ENV_HI - ENV_GATE + 1e-12), 0.0, 1.0))
                base_speed = SPEED_MIN + (SPEED_MAX - SPEED_MIN) * (t_env * t_env)

                if CRACK_ENABLE:
                    env_fast = (1.0 - a_fast) * env_fast + a_fast * env
                    env_slow = (1.0 - a_slow) * env_slow + a_slow * env
                    ratio    = env_fast / (env_slow + 1e-9)
                else:
                    ratio = 0.0

                if (not crack_armed) and (ratio < CRACK_RATIO_OFF):
                    crack_armed = True

                if state == "CRACK":
                    x += CRACK_SPEED * math.cos(crack_theta) * DT
                    y += CRACK_SPEED * math.sin(crack_theta) * DT
                    x  = float(np.clip(x, PAD_X, W - PAD_X))
                    y  = float(np.clip(y, PAD_Y, H - PAD_Y))
                    on_move(x, y)
                    crack_remain -= DT
                    if crack_remain <= 0.0:
                        on_crack_point(x, y)
                        state              = "GROW"
                        crack_cooldown     = CRACK_COOLDOWN_S
                        theta              = crack_theta + random.uniform(-0.6, 0.6)
                    time.sleep(DT * 0.5)
                    continue

                elapsed    = time.time() - vine_start_t
                startup_ok = elapsed > CRACK_STARTUP_DELAY_S

                if vine_dist_traveled >= vine_max_dist:
                    break
                if vine_dist_traveled >= VINE_MIN_DIST and elapsed >= VINE_TIMEOUT_S:
                    break

                if crack_cooldown > 0.0:
                    crack_cooldown -= DT

                if CRACK_ENABLE and crack_armed and startup_ok and crack_cooldown <= 0.0 and ratio > CRACK_RATIO_ON:
                    crack_armed        = False
                    state              = "CRACK"
                    in_crack_flag[0]   = True
                    angle_offset       = random.uniform(CRACK_ANGLE_MIN, CRACK_ANGLE_MAX)
                    side_sign          = random.choice((-1, 1))
                    crack_theta        = theta + side_sign * angle_offset
                    crack_len          = random.uniform(CRACK_LEN_MIN, CRACK_LEN_MAX)
                    crack_remain       = crack_len / CRACK_SPEED
                    time.sleep(DT * 0.5)
                    continue

                ctrl = y_audio
                if bpf is not None:
                    ctrl = bpf.process(ctrl)
                ctrl      = ctrl_lpf.process(ctrl)
                c         = float(np.mean(ctrl))
                ctrl_s    = 0.85 * ctrl_s + 0.15 * c
                ctrl_norm = ctrl_s / (env_s + 1e-6)

                drift_align = wrap_pi(drift_theta - theta)
                fx_w, fy_w  = wall_force(x, y)
                wall_mag    = math.hypot(fx_w, fy_w)
                wall_align  = wrap_pi(math.atan2(fy_w, fx_w) - theta) if wall_mag > 1e-9 else 0.0

                weave_phase = (weave_phase + 2.0 * math.pi * WEAVE_FREQ_HZ * DT) % (2.0 * math.pi)
                weave_amp   = lerp(WEAVE_AMP_MIN, WEAVE_AMP_MAX, t_env)
                weave       = weave_amp * math.sin(weave_phase)

                eta       = 0.985 * eta + 0.015 * random.uniform(-1.0, 1.0)
                noise_amp = NOISE_BASE + NOISE_GAIN * t_env

                omega_target = (K_DRIFT * drift_align
                                + 3.0 * (wall_mag / (WALL_FORCE + 1e-9)) * wall_align
                                + weave
                                + K_AUDIO * ctrl_norm
                                + noise_amp * eta)
                omega = float(np.clip(OMEGA_DAMP * omega + (1.0 - OMEGA_DAMP) * omega_target,
                                      -OMEGA_MAX, OMEGA_MAX))

                theta += omega * DT
                x     += base_speed * math.cos(theta) * DT
                y     += base_speed * math.sin(theta) * DT
                x      = float(np.clip(x, PAD_X, W - PAD_X))
                y      = float(np.clip(y, PAD_Y, H - PAD_Y))

                on_move(x, y)

                step = math.hypot(x - last_x, y - last_y)
                vine_dist_traveled += step
                last_x, last_y = x, y
                blue_pts.append((x, y))

                time.sleep(DT * 0.5)

            if stop_event.is_set():
                break

            bp = _pick_branch_point(blue_pts) or (x, y)
            on_branch(bp[0], bp[1])
            _deposit_blue_pts(blue_pts)
            vine_x      = bp[0]
            vine_y      = bp[1]
            drift_theta = _next_drift(bp[0], bp[1], drift_theta)
            vine_num   += 1

    on_stop()


# ===================== DRAWING MODE =====================
def run(machine, stop_event: threading.Event, pause_event: threading.Event) -> None:
    """
    Called by the GUI.
    Within a vine: pen never lifts (blue and red are one continuous path).
    Between vines: pen lifts, rapids to branch point, pen down, new vine starts.
    """
    from machine import draw_to_machine, DRAW_FEED

    last_pos = [None]

    def _send(x, y, rapid=False):
        lp = last_pos[0]
        mx, my = draw_to_machine(x, y)
        if lp is None:
            machine.send_gcode(f"G0 X{mx:.3f} Y{my:.3f}")
            machine.pen_down()
            last_pos[0] = (x, y)
            return
        if math.hypot(x - lp[0], y - lp[1]) >= DRAW_MIN_SEGMENT:
            cmd = "G0" if rapid else "G1"
            feed = f" F{DRAW_FEED}" if not rapid else ""
            machine.send_gcode(f"{cmd} X{mx:.3f} Y{my:.3f}{feed}")
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
        pass   # crack tip — path already drawn via on_move

    def on_branch(bx, by):
        """Lift pen, rapid to branch point, pen down — start new vine."""
        if stop_event.is_set():
            return
        machine.pen_up()
        mx, my = draw_to_machine(bx, by)
        machine.send_gcode(f"G0 X{mx:.3f} Y{my:.3f}")
        machine.pen_down()
        last_pos[0] = (bx, by)

    def on_stop():
        machine.pen_up()

    _vine_loop(on_move, on_stop, on_crack_point, on_branch, stop_event)


# ===================== ENTRY POINT =====================
if __name__ == "__main__":
    run_simulation()
